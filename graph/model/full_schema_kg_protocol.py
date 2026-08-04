#!/usr/bin/env python3
"""Full-schema NEMSIS KG builder and leakage-safe protocol trainer.

This is intentionally independent from train_kg_protocol.py.

Stages:
  build  - stream NEMSIS ASCII tables into a normalized SQLite KG
  train  - train a relation-aware neighborhood encoder over the persisted KG
  all    - run both stages

The graph stores all requested relations. By default, prediction only consumes
relations known at protocol-selection time; target/post-treatment relations stay
available for train-only KG representation learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEP = "~|~"

NODE_TYPES = (
    "PCR", "SymptomCode", "ImpressionCode", "ICDCategory", "Protocol",
    "ProtocolCategory", "AgeGroup", "DispatchComplaint", "VitalObservation",
    "ECGRhythm", "Medication", "MedicationAdministration", "Procedure",
    "ProcedureEvent", "InjuryCause", "TraumaCriterion", "HospitalDiagnosis",
    "Region",
)

# True means available to the default protocol-prediction query encoder.
RELATIONS = {
    "has_primary_symptom": True,
    "has_additional_symptom": True,
    "has_primary_impression": True,
    "has_secondary_impression": True,
    "used_protocol": False,                 # target
    "belongs_to": True,
    "applies_to": True,
    "is_a": True,
    "has_age_group": True,                  # needed to connect PCR and AgeGroup
    "has_dispatch_complaint": True,
    "has_initial_vital": True,
    "has_ecg_rhythm": True,
    "has_injury_cause": True,
    "meets_trauma_criterion": True,
    "located_in": True,
    "received_medication": False,           # generally post-decision
    "administers": False,                   # MedicationAdministration -> Medication
    "underwent_procedure": False,           # generally post-decision
    "procedure_type": False,                # ProcedureEvent -> Procedure
    "hospital_diagnosis": False,            # future outcome
}

QUERY_RELATIONS = tuple(k for k, available in RELATIONS.items() if available)
POST_RELATIONS = ("received_medication", "underwent_procedure", "hospital_diagnosis")
QUERY_NODE_TYPES = ("SymptomCode", "ImpressionCode", "AgeGroup", "DispatchComplaint", "VitalObservation", "ECGRhythm", "InjuryCause", "TraumaCriterion", "Region")


def cells(line: str) -> list[str]:
    return [x.strip().strip("'").strip('"') for x in line.rstrip("\r\n").split(SEP)]


def grouped_split(signature: str, seed: int) -> str:
    bucket = int.from_bytes(
        hashlib.blake2b(f"{seed}|{signature}".encode(), digest_size=8).digest(),
        "little",
    ) % 10
    return "train" if bucket < 7 else ("validation" if bucket == 7 else "test")


def age_group(value: str) -> str:
    try:
        age = float(value)
    except ValueError:
        return "Unknown"
    if age < 0.08:
        return "Neonate"
    if age < 1:
        return "Infant"
    if age < 12:
        return "Child"
    if age < 18:
        return "Adolescent"
    if age < 65:
        return "Adult"
    return "OlderAdult"


def icd_category(code: str) -> str:
    stem = code.split(".")[0].upper()
    return stem[:3] if len(stem) >= 3 else stem


def numeric(value: str):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


class KGWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS node(
              id INTEGER PRIMARY KEY, type TEXT NOT NULL, key TEXT NOT NULL,
              attrs TEXT NOT NULL DEFAULT '{}', UNIQUE(type,key));
            CREATE TABLE IF NOT EXISTS edge(
              src INTEGER NOT NULL, relation TEXT NOT NULL, dst INTEGER NOT NULL,
              split TEXT NOT NULL, prediction_available INTEGER NOT NULL,
              attrs TEXT NOT NULL DEFAULT '{}',
              UNIQUE(src,relation,dst,split));
            CREATE TABLE IF NOT EXISTS pcr(
              pcr_key TEXT PRIMARY KEY, node_id INTEGER NOT NULL, split TEXT NOT NULL,
              label TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS edge_src_rel ON edge(src,relation);
            CREATE INDEX IF NOT EXISTS edge_dst_rel ON edge(dst,relation);
            CREATE INDEX IF NOT EXISTS pcr_split ON pcr(split);
            """
        )
        self.node_cache: dict[tuple[str, str], int] = {}

    def node(self, kind: str, key: str, attrs=None) -> int:
        if kind not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {kind}")
        key = key.strip() or "__MISSING__"
        cache_key = (kind, key)
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]
        attrs_json = json.dumps(attrs or {}, separators=(",", ":"))
        self.db.execute(
            "INSERT OR IGNORE INTO node(type,key,attrs) VALUES(?,?,?)",
            (kind, key, attrs_json),
        )
        row = self.db.execute(
            "SELECT id FROM node WHERE type=? AND key=?", cache_key
        ).fetchone()
        self.node_cache[cache_key] = row[0]
        return row[0]

    def edge(self, src: int, relation: str, dst: int, split: str, attrs=None):
        if relation not in RELATIONS:
            raise ValueError(f"Unknown relation: {relation}")
        self.db.execute(
            "INSERT OR IGNORE INTO edge VALUES(?,?,?,?,?,?)",
            (src, relation, dst, split, int(RELATIONS[relation]),
             json.dumps(attrs or {}, separators=(",", ":"))),
        )

    def commit(self):
        self.db.commit()


def scan_table(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = cells(next(f))
        for line in f:
            values = cells(line)
            if len(values) == len(header):
                yield dict(zip(header, values))


def build_kg(args):
    kg_path = Path(args.kg)
    if kg_path.exists() and not args.rebuild:
        raise FileExistsError(f"{kg_path} exists; pass --rebuild to replace it")
    if kg_path.exists():
        kg_path.unlink()
    writer = KGWriter(kg_path)
    wanted: dict[str, tuple[int, str, str]] = {}
    label_path = Path(args.labels)

    # Core PCR, four clinical relations, target, and ICD category hierarchy.
    for row in scan_table(label_path):
        fields = [
            row["PrimarySymptomCode"].split(),
            row["PrimaryImpressionCode"].split(),
            row["AdditionalSymptomCode"].split(),
            row["SecondaryImpressionCode"].split(),
        ]
        signature = "\x1e".join(" ".join(v) for v in fields)
        split = grouped_split(signature, args.seed)
        key, label = row["PcrKey"], row["ProtocolIds"]
        pcr = writer.node("PCR", key)
        wanted[key] = (pcr, split, label)
        writer.db.execute(
            "INSERT INTO pcr VALUES(?,?,?,?)", (key, pcr, split, label)
        )
        specs = (
            ("has_primary_symptom", "SymptomCode", fields[0]),
            ("has_primary_impression", "ImpressionCode", fields[1]),
            ("has_additional_symptom", "SymptomCode", fields[2]),
            ("has_secondary_impression", "ImpressionCode", fields[3]),
        )
        for relation, kind, codes in specs:
            for code in dict.fromkeys(codes or ["__MISSING__"]):
                code_node = writer.node(kind, code)
                writer.edge(pcr, relation, code_node, split)
                if code != "__MISSING__":
                    category = writer.node("ICDCategory", icd_category(code))
                    writer.edge(code_node, "is_a", category, split)
        protocol = writer.node("Protocol", label)
        writer.edge(pcr, "used_protocol", protocol, split)
        # A category node always exists. Supply a mapping JSON to replace UNMAPPED.
        category_name = args.protocol_category_map.get(label, "UNMAPPED")
        category = writer.node("ProtocolCategory", category_name)
        writer.edge(protocol, "belongs_to", category, split)
    writer.commit()

    raw = Path(args.raw_dir)
    wanted_keys = set(wanted)

    # Age/region and observed train-only Protocol applicability.
    protocol_age_pairs = set()
    for row in scan_table(raw / "ComputedElements.txt"):
        key = row["PcrKey"]
        if key not in wanted_keys:
            continue
        pcr, split, label = wanted[key]
        group = age_group(row["ageinyear"])
        age_node = writer.node("AgeGroup", group)
        writer.edge(pcr, "has_age_group", age_node, split)
        if split == "train":
            protocol_age_pairs.add((label, group))
        for region_kind in ("USCensusRegion", "USCensusDivision", "NasemsoRegion", "Urbanicity"):
            value = row[region_kind]
            region = writer.node("Region", f"{region_kind}:{value}")
            writer.edge(pcr, "located_in", region, split)
    for label, group in protocol_age_pairs:
        writer.edge(
            writer.node("Protocol", label), "applies_to",
            writer.node("AgeGroup", group), "train",
        )
    writer.commit()

    # Dispatch complaint.
    for row in scan_table(raw / "Pub_PCRevents.txt"):
        key = row["PcrKey"]
        if key in wanted:
            pcr, split, _ = wanted[key]
            complaint = writer.node("DispatchComplaint", row["eDispatch_01"])
            writer.edge(pcr, "has_dispatch_complaint", complaint, split)
    writer.commit()

    # Keep exactly the earliest lexicographically valid vital timestamp per PCR.
    earliest: dict[str, tuple[str, int]] = {}
    vital_cols = (
        "eVitals_06", "eVitals_10", "eVitals_12", "eVitals_14", "eVitals_16",
        "eVitals_18", "eVitals_27", "eVitals_29", "eVitals_30", "eVitals_31",
        "eVitals_19", "eVitals_20", "eVitals_21",
    )
    for row in scan_table(raw / "FACTPCRVITAL.txt"):
        key = row["PcrKey"]
        if key not in wanted:
            continue
        timestamp = row["eVitals_01"]
        order = timestamp if timestamp not in ("", "Not Applicable", "Not Recorded") else "9999"
        if key in earliest and order >= earliest[key][0]:
            continue
        attrs = {c: numeric(row[c]) for c in vital_cols}
        attrs["time"] = timestamp
        vital = writer.node("VitalObservation", row["PcrVitalKey"], attrs)
        earliest[key] = (order, vital)
    initial_vital_ids = set()
    for key, (_, vital) in earliest.items():
        pcr, split, _ = wanted[key]
        writer.edge(pcr, "has_initial_vital", vital, split)
        initial_vital_ids.add(vital)
    writer.commit()

    # ECG group uses VitalKey. Only connect ECGs belonging to selected initial vitals.
    vital_key_to_id = {
        key: node_id for key, (_, node_id) in earliest.items()
    }
    # Resolve PcrVitalKey -> node id directly from node table to avoid assuming key=PCR.
    selected_vital_keys = {
        row[0]: row[1] for row in writer.db.execute(
            "SELECT key,id FROM node WHERE type='VitalObservation'"
        )
    }
    for row in scan_table(raw / "PCRVITALECGGROUP.txt"):
        vital = selected_vital_keys.get(row["VitalKey"])
        if vital is None:
            continue
        rhythm = writer.node("ECGRhythm", row["eVitals_03"])
        # Attach to PCR so query batching does not require two-hop SQL traversal.
        pcr_row = writer.db.execute(
            "SELECT src,split FROM edge WHERE relation='has_initial_vital' AND dst=?",
            (vital,),
        ).fetchone()
        if pcr_row:
            writer.edge(pcr_row[0], "has_ecg_rhythm", rhythm, pcr_row[1])
    writer.commit()

    simple_tables = (
        ("FACTPCRCAUSEOFINJURY.txt", "eInjury_01", "InjuryCause", "has_injury_cause"),
        ("FACTPCRTRAUMACRITERIA.txt", "eInjury_03", "TraumaCriterion", "meets_trauma_criterion"),
        ("FactPcreOutcomeHospDiag.txt", "eOutcome_13", "HospitalDiagnosis", "hospital_diagnosis"),
    )
    for filename, column, kind, relation in simple_tables:
        for row in scan_table(raw / filename):
            key = row["PcrKey"]
            if key in wanted:
                pcr, split, _ = wanted[key]
                writer.edge(pcr, relation, writer.node(kind, row[column]), split)
        writer.commit()

    for row in scan_table(raw / "FACTPCRMEDICATION.txt"):
        key = row["PcrKey"]
        if key not in wanted:
            continue
        pcr, split, _ = wanted[key]
        admin = writer.node(
            "MedicationAdministration", row["PcrMedicationKey"],
            {k: row[k] for k in ("eMedications_01", "eMedications_04",
                                 "eMedications_05", "eMedications_06",
                                 "eMedications_07")},
        )
        medication = writer.node(
            "Medication", row["eMedications_03"],
            {"description": row["eMedications_03Descr"]},
        )
        writer.edge(pcr, "received_medication", admin, split)
        writer.edge(admin, "administers", medication, split)
    writer.commit()

    for row in scan_table(raw / "FACTPCRPROCEDURE.txt"):
        key = row["PcrKey"]
        if key not in wanted:
            continue
        pcr, split, _ = wanted[key]
        event = writer.node(
            "ProcedureEvent", row["PcrProcedureKey"],
            {k: row[k] for k in ("eProcedures_01", "eProcedures_05",
                                 "eProcedures_06", "eProcedures_08")},
        )
        procedure = writer.node("Procedure", row["eProcedures_03"])
        writer.edge(pcr, "underwent_procedure", event, split)
        writer.edge(event, "procedure_type", procedure, split)
    writer.commit()
    writer.db.execute("ANALYZE")
    writer.commit()
    print_schema_stats(writer.db)
    writer.db.close()


def print_schema_stats(db):
    print("nodes:")
    for row in db.execute("SELECT type,count(*) FROM node GROUP BY type ORDER BY type"):
        print(f"  {row[0]}: {row[1]}")
    print("edges:")
    for row in db.execute(
        "SELECT relation,count(*) FROM edge GROUP BY relation ORDER BY relation"
    ):
        print(f"  {row[0]}: {row[1]}")


class PCRNeighbors(Dataset):
    """Materialize leakage-safe 1-hop PCR neighborhoods from SQLite."""
    def __init__(self, kg: str, split: str, entity_to_id, relation_to_id,
                 include_post_treatment=False):
        db = sqlite3.connect(kg)
        allowed = list(QUERY_RELATIONS)
        if include_post_treatment:
            allowed += list(POST_RELATIONS)
        placeholders = ",".join("?" * len(allowed))
        rows = db.execute(
            f"""
            SELECT p.pcr_key,p.label,e.relation,n.type,n.key,n.attrs
            FROM pcr p
            LEFT JOIN edge e ON e.src=p.node_id AND e.relation IN ({placeholders})
            LEFT JOIN node n ON n.id=e.dst
            WHERE p.split=?
            ORDER BY p.pcr_key
            """,
            (*allowed, split),
        )
        grouped = {}
        for pcr_key, label, relation, kind, key, attrs in rows:
            item = grouped.setdefault(pcr_key, {"label": label, "neighbors": [], "vital": []})
            if relation is None:
                continue
            entity = entity_to_id.get(f"{kind}:{key}", entity_to_id["__UNK__"])
            item["neighbors"].append((relation_to_id[relation], entity))
            if kind == "VitalObservation":
                values = json.loads(attrs)
                vital_keys = (
                    "eVitals_06", "eVitals_10", "eVitals_12", "eVitals_14",
                    "eVitals_16", "eVitals_18", "eVitals_27", "eVitals_29",
                    "eVitals_30", "eVitals_31", "eVitals_19", "eVitals_20",
                    "eVitals_21",
                )
                item["vital"] = [
                    float(values[k]) if values.get(k) is not None else float("nan")
                    for k in vital_keys
                ]
        db.close()
        self.items = list(grouped.values())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items, label_to_id):
    width = max(max(len(x["neighbors"]), 1) for x in items)
    relations = torch.zeros(len(items), width, dtype=torch.long)
    entities = torch.zeros(len(items), width, dtype=torch.long)
    mask = torch.zeros(len(items), width, dtype=torch.bool)
    vitals = torch.full((len(items), 13), float("nan"))
    labels = torch.tensor([label_to_id[x["label"]] for x in items])
    for i, item in enumerate(items):
        for j, (relation, entity) in enumerate(item["neighbors"]):
            relations[i, j], entities[i, j], mask[i, j] = relation, entity, True
        if item["vital"]:
            vitals[i] = torch.tensor(item["vital"])
    return relations, entities, mask, vitals, labels


class FullSchemaModel(nn.Module):
    def __init__(self, n_entities, n_relations, n_labels, dim=128, dropout=0.2):
        super().__init__()
        self.entity = nn.Embedding(n_entities, dim)
        self.relation = nn.Embedding(n_relations, dim)
        self.query = nn.Parameter(torch.empty(dim))
        self.vital = nn.Sequential(nn.Linear(26, dim), nn.ReLU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(dim)
        )
        self.protocol = nn.Embedding(n_labels, dim)
        self.bias = nn.Parameter(torch.zeros(n_labels))
        for emb in (self.entity, self.relation, self.protocol):
            nn.init.xavier_uniform_(emb.weight)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, relations, entities, mask, vitals):
        h = self.entity(entities) + self.relation(relations)
        score = (h * self.query).sum(-1).masked_fill(~mask, -1e4)
        pooled = (torch.softmax(score, -1)[..., None] * h).sum(1)
        missing = torch.isnan(vitals).float()
        filled = torch.nan_to_num(vitals)
        context = self.vital(torch.cat([filled, missing], 1))
        case = nn.functional.normalize(self.fusion(torch.cat([pooled, context], 1)), dim=-1)
        protocol = nn.functional.normalize(self.protocol.weight, dim=-1)
        return case @ protocol.T * math.sqrt(case.shape[1]) + self.bias


def metrics(y, logits):
    order = np.argsort(-logits, axis=1)
    pred = order[:, 0]
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )
    result = {"top1": float(np.mean(pred == y)), "accuracy": float(np.mean(pred == y))}
    for k in (3, 5):
        result[f"top{k}"] = float(np.mean(np.any(order[:, :k] == y[:, None], axis=1)))
    result.update(macro_precision=float(p), macro_recall=float(r), macro_f1=float(f))
    return result


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, scores = [], []
    for rel, ent, mask, vital, y in loader:
        scores.append(model(rel.to(device), ent.to(device), mask.to(device),
                            vital.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return metrics(np.concatenate(ys), np.concatenate(scores))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    db = sqlite3.connect(args.kg)
    labels = [r[0] for r in db.execute("SELECT DISTINCT label FROM pcr ORDER BY label")]
    # Only neighbor concepts need embeddings; PCR/event node embeddings are not used.
    kinds = QUERY_NODE_TYPES + (("MedicationAdministration", "ProcedureEvent", "HospitalDiagnosis") if args.include_post_treatment else ())
    placeholders = ",".join("?" * len(kinds))
    entity_keys = [
        f"{kind}:{key}" for kind, key in db.execute(
            f"SELECT type,key FROM node WHERE type IN ({placeholders})", kinds
        )
    ]
    db.close()
    entity_to_id = {"__PAD__": 0, "__UNK__": 1}
    entity_to_id.update({key: i + 2 for i, key in enumerate(entity_keys)})
    model_relations = list(QUERY_RELATIONS)
    if args.include_post_treatment:
        model_relations += list(POST_RELATIONS)
    relation_to_id = {r: i + 1 for i, r in enumerate(model_relations)}
    relation_to_id["__PAD__"] = 0
    label_to_id = {x: i for i, x in enumerate(labels)}
    datasets = [
        PCRNeighbors(args.kg, split, entity_to_id, relation_to_id,
                     args.include_post_treatment)
        for split in ("train", "validation", "test")
    ]
    loaders = [
        DataLoader(
            ds, batch_size=args.batch_size, shuffle=(i == 0),
            num_workers=0, collate_fn=lambda x: collate(x, label_to_id),
        )
        for i, ds in enumerate(datasets)
    ]
    model = FullSchemaModel(
        len(entity_to_id), len(relation_to_id), len(labels), args.dim, args.dropout
    ).to(device)
    counts = np.bincount(
        [label_to_id[x["label"]] for x in datasets[0].items], minlength=len(labels)
    )
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights /= weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device).float())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for rel, ent, mask, vital, y in loaders[0]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(rel.to(device), ent.to(device), mask.to(device), vital.to(device))
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        val = evaluate(model, loaders[1], device)
        print(json.dumps({"epoch": epoch, "loss": total / count, **val}))
        if val["macro_f1"] > best:
            best, stale = val["macro_f1"], 0
            torch.save(
                {"state_dict": model.state_dict(), "labels": labels,
                 "entity_to_id": entity_to_id, "relation_to_id": relation_to_id,
                 "args": vars(args)},
                output / "best_full_schema_model.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(
        output / "best_full_schema_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    test_result = evaluate(model, loaders[2], device)
    report = {"best_validation_macro_f1": best, "test": test_result}
    (output / "full_schema_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("test", json.dumps(test_result))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("build", "train", "all"), default="all")
    p.add_argument("--labels", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--kg", default="cache/nemsis_full_schema.sqlite")
    p.add_argument("--output-dir", default="outputs/full_schema_kg")
    p.add_argument("--protocol-category-json")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--include-post-treatment", action="store_true",
                   help="Ablation only: exposes medication/procedure/outcome to predictor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    args.protocol_category_map = {}
    if args.protocol_category_json:
        args.protocol_category_map = json.loads(
            Path(args.protocol_category_json).read_text(encoding="utf-8")
        )
    return args


if __name__ == "__main__":
    options = parse_args()
    if options.stage in ("build", "all"):
        build_kg(options)
    if options.stage in ("train", "all"):
        train(options)

