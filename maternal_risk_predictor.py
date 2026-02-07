import base64
import hashlib
import os
import secrets
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RISK_COLUMN = "Risk Level"
FEATURE_COLUMNS = [
	"Age",
	"Systolic BP",
	"Diastolic",
	"BS",
	"Body Temp",
	"BMI",
	"Previous Complications",
	"Preexisting Diabetes",
	"Gestational Diabetes",
	"Mental Health",
	"Heart Rate",
]

DIABETES_TARGET = "diagnosed_diabetes"
DIABETES_EXCLUDE_COLUMNS = {"diabetes_stage", "diabetes_risk_score"}
DIABETES_CATEGORICAL_COLUMNS = [
	"gender",
	"ethnicity",
	"education_level",
	"income_level",
	"employment_status",
	"smoking_status",
]

RANGE_RULES = {
	"Age": (10, 70),
	"Systolic BP": (60, 200),
	"Diastolic": (40, 140),
	"BS": (2, 30),
	"Body Temp": (95, 105),
	"BMI": (10, 60),
	"Heart Rate": (40, 140),
	"Previous Complications": (0, 1),
	"Preexisting Diabetes": (0, 1),
	"Gestational Diabetes": (0, 1),
	"Mental Health": (0, 1),
}

USER_STORE = {
	"doctor1": {"password": "doctor123", "role": "doctor"},
	"viewer1": {"password": "viewer123", "role": "viewer"},
}

DB_PATH = os.path.join(os.path.dirname(__file__), "maternal_app.db")
PRIORITY_TAGS = ["None", "Low", "Medium", "High", "Urgent"]


def load_csv(file) -> pd.DataFrame:
	if file is None:
		return pd.read_csv("Dataset - Updated.csv")
	return pd.read_csv(file)


def load_diabetes_csv(file) -> pd.DataFrame:
	if file is None:
		return pd.read_csv("diabetes_dataset.csv")
	return pd.read_csv(file)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
	df = df.copy()
	df.columns = [col.strip() for col in df.columns]

	report = {
		"rows_start": len(df),
		"missing_target": 0,
		"outlier_fixes": {},
		"missing_after_clean": {},
	}

	if RISK_COLUMN not in df.columns:
		raise ValueError(f"Missing '{RISK_COLUMN}' column.")

	df[RISK_COLUMN] = df[RISK_COLUMN].astype(str).str.strip()
	missing_target_mask = df[RISK_COLUMN].eq("") | df[RISK_COLUMN].isna()
	report["missing_target"] = int(missing_target_mask.sum())
	df = df.loc[~missing_target_mask].reset_index(drop=True)

	for col in FEATURE_COLUMNS:
		if col not in df.columns:
			df[col] = np.nan
		df[col] = pd.to_numeric(df[col], errors="coerce")

	for col, (low, high) in RANGE_RULES.items():
		if col not in df.columns:
			continue
		outlier_mask = (df[col] < low) | (df[col] > high)
		report["outlier_fixes"][col] = int(outlier_mask.sum())
		df.loc[outlier_mask, col] = np.nan

	report["missing_after_clean"] = df[FEATURE_COLUMNS].isna().sum().to_dict()
	report["rows_end"] = len(df)
	return df, report


def clean_diabetes_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
	df = df.copy()
	df.columns = [col.strip() for col in df.columns]

	report = {
		"rows_start": len(df),
		"missing_target": 0,
		"missing_after_clean": {},
		"rows_end": 0,
	}

	if DIABETES_TARGET not in df.columns:
		raise ValueError(f"Missing '{DIABETES_TARGET}' column.")

	df[DIABETES_TARGET] = pd.to_numeric(df[DIABETES_TARGET], errors="coerce")
	missing_target_mask = df[DIABETES_TARGET].isna()
	report["missing_target"] = int(missing_target_mask.sum())
	df = df.loc[~missing_target_mask].reset_index(drop=True)

	available_categoricals = [
		col for col in DIABETES_CATEGORICAL_COLUMNS if col in df.columns
	]
	feature_columns = [
		col
		for col in df.columns
		if col not in {DIABETES_TARGET, *DIABETES_EXCLUDE_COLUMNS}
	]
	for col in feature_columns:
		if col in available_categoricals:
			continue
		df[col] = pd.to_numeric(df[col], errors="coerce")

	report["missing_after_clean"] = df[feature_columns].isna().sum().to_dict()
	report["rows_end"] = len(df)
	return df, report


@st.cache_data(show_spinner=False)
def prepare_data(file) -> tuple[pd.DataFrame, dict]:
	df = load_csv(file)
	return clean_data(df)


@st.cache_data(show_spinner=False)
def prepare_diabetes_data(file) -> tuple[pd.DataFrame, dict]:
	df = load_diabetes_csv(file)
	return clean_diabetes_data(df)


def build_model() -> Pipeline:
	return Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			(
				"classifier",
				RandomForestClassifier(
					n_estimators=300,
					random_state=42,
					class_weight="balanced",
				),
			),
		]
	)


def build_diabetes_model(
	numeric_features: list[str],
	categorical_features: list[str],
) -> Pipeline:
	preprocess = ColumnTransformer(
		transformers=[
			(
				"numeric",
				SimpleImputer(strategy="median"),
				numeric_features,
			),
			(
				"categorical",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						(
							"onehot",
							OneHotEncoder(handle_unknown="ignore"),
						),
					]
				),
				categorical_features,
			),
		],
		remainder="drop",
	)
	return Pipeline(
		steps=[
			("preprocess", preprocess),
			(
				"classifier",
				RandomForestClassifier(
					n_estimators=300,
					random_state=42,
					class_weight="balanced",
				),
			),
		]
	)


def get_priority_scores(model: Pipeline, feature_frame: pd.DataFrame) -> np.ndarray:
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(feature_frame)
		class_labels = list(model.named_steps["classifier"].classes_)
		if "High" in class_labels:
			high_index = class_labels.index("High")
			return proba[:, high_index]
		return proba.max(axis=1)
	return model.predict(feature_frame)


def get_binary_risk_scores(
	model: Pipeline,
	feature_frame: pd.DataFrame,
	positive_label: int = 1,
) -> np.ndarray:
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(feature_frame)
		class_labels = list(model.named_steps["classifier"].classes_)
		if positive_label in class_labels:
			pos_index = class_labels.index(positive_label)
			return proba[:, pos_index]
		return proba.max(axis=1)
	return model.predict(feature_frame)


def build_patient_summary(patient_row: pd.Series, reference_df: pd.DataFrame) -> list[str]:
	medians = reference_df[FEATURE_COLUMNS].median()
	iqr = reference_df[FEATURE_COLUMNS].quantile(0.75) - reference_df[FEATURE_COLUMNS].quantile(0.25)
	iqr = iqr.replace(0, 1)
	z_scores = (patient_row[FEATURE_COLUMNS] - medians) / iqr
	top_features = z_scores.abs().sort_values(ascending=False).head(3).index
	summary = []
	for feature in top_features:
		value = patient_row.get(feature, np.nan)
		summary.append(f"{feature}: {value}")
	return summary


def svg_to_data_uri(svg_text: str) -> str:
	encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
	return f"data:image/svg+xml;base64,{encoded}"


def render_medical_gallery() -> None:
	stethoscope_svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="240" height="160" viewBox="0 0 240 160">
  <rect width="240" height="160" rx="18" fill="#f4f7fb"/>
  <rect x="14" y="14" width="212" height="132" rx="14" fill="#ffffff"/>
  <circle cx="70" cy="56" r="16" fill="#0f9b88"/>
  <circle cx="170" cy="56" r="16" fill="#5b7cfa"/>
  <path d="M70 72v26c0 22 14 36 50 36s50-14 50-36V72" fill="none" stroke="#0f3d33" stroke-width="8" stroke-linecap="round"/>
  <circle cx="120" cy="124" r="14" fill="#ff7a59"/>
  <path d="M120 110v-14" stroke="#ffffff" stroke-width="6" stroke-linecap="round"/>
  <path d="M112 117h16" stroke="#ffffff" stroke-width="6" stroke-linecap="round"/>
</svg>
"""

	heartbeat_svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="240" height="160" viewBox="0 0 240 160">
  <rect width="240" height="160" rx="18" fill="#f7f1ec"/>
  <rect x="14" y="14" width="212" height="132" rx="14" fill="#ffffff"/>
  <path d="M32 92h28l14-24 16 44 18-36 14 16h44" fill="none" stroke="#ff7a59" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="188" cy="56" r="18" fill="#0f9b88"/>
  <path d="M180 56h16" stroke="#ffffff" stroke-width="6" stroke-linecap="round"/>
  <path d="M188 48v16" stroke="#ffffff" stroke-width="6" stroke-linecap="round"/>
</svg>
"""

	stethoscope_uri = svg_to_data_uri(stethoscope_svg)
	heartbeat_uri = svg_to_data_uri(heartbeat_svg)
	st.markdown(
		f"""
		<div class="image-grid">
			<div class="image-card">
				<img src="{stethoscope_uri}" alt="Stethoscope illustration" />
				<div>
					<h5>Care-ready workflows</h5>
					<p>Consistent intake, clear triage, and clinician-friendly views.</p>
				</div>
			</div>
			<div class="image-card">
				<img src="{heartbeat_uri}" alt="Heartbeat illustration" />
				<div>
					<h5>Signal-driven insights</h5>
					<p>Use risk signals to guide follow-ups and care prioritization.</p>
				</div>
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
	conn = sqlite3.connect(DB_PATH, check_same_thread=False)
	conn.row_factory = sqlite3.Row
	return conn


def hash_password(password: str, salt: str) -> str:
	return hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()


def init_db() -> None:
	conn = get_db_connection()
	cursor = conn.cursor()
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS users (
			username TEXT PRIMARY KEY,
			password_hash TEXT NOT NULL,
			salt TEXT NOT NULL,
			role TEXT NOT NULL
		)
		"""
	)
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS notes (
			patient_id INTEGER PRIMARY KEY,
			note TEXT NOT NULL,
			priority_tag TEXT NOT NULL DEFAULT 'None',
			updated_at TEXT NOT NULL,
			updated_by TEXT NOT NULL
		)
		"""
	)
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS note_history (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			patient_id INTEGER NOT NULL,
			note TEXT NOT NULL,
			priority_tag TEXT NOT NULL DEFAULT 'None',
			created_at TEXT NOT NULL,
			created_by TEXT NOT NULL
		)
		"""
	)
	conn.commit()

	ensure_column(conn, "notes", "priority_tag", "TEXT", "None")
	ensure_column(conn, "note_history", "priority_tag", "TEXT", "None")

	count = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
	if count == 0:
		for username, record in USER_STORE.items():
			salt = secrets.token_hex(16)
			password_hash = hash_password(record["password"], salt)
			cursor.execute(
				"INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
				(username, password_hash, salt, record["role"]),
			)
		conn.commit()


def ensure_column(
	conn: sqlite3.Connection,
	table_name: str,
	column_name: str,
	column_type: str,
	default_value: str,
) -> None:
	columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
	column_names = {row["name"] for row in columns}
	if column_name in column_names:
		return
	conn.execute(
		f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
	)
	conn.execute(
		f"UPDATE {table_name} SET {column_name} = ? WHERE {column_name} IS NULL",
		(default_value,),
	)
	conn.commit()


def verify_user(username: str, password: str) -> str | None:
	conn = get_db_connection()
	row = conn.execute(
		"SELECT password_hash, salt, role FROM users WHERE username = ?", (username,)
	).fetchone()
	if row is None:
		return None
	if hash_password(password, row["salt"]) == row["password_hash"]:
		return row["role"]
	return None


def get_notes_map(patient_ids: list[int]) -> dict[int, str]:
	if not patient_ids:
		return {}
	placeholders = ",".join("?" for _ in patient_ids)
	query = f"SELECT patient_id, note FROM notes WHERE patient_id IN ({placeholders})"
	conn = get_db_connection()
	rows = conn.execute(query, patient_ids).fetchall()
	return {row["patient_id"]: row["note"] for row in rows}


def get_tags_map(patient_ids: list[int]) -> dict[int, str]:
	if not patient_ids:
		return {}
	placeholders = ",".join("?" for _ in patient_ids)
	query = f"SELECT patient_id, priority_tag FROM notes WHERE patient_id IN ({placeholders})"
	conn = get_db_connection()
	rows = conn.execute(query, patient_ids).fetchall()
	return {row["patient_id"]: row["priority_tag"] for row in rows}



def get_latest_note_record(patient_id: int) -> dict:
	conn = get_db_connection()
	row = conn.execute(
		"""
		SELECT note, priority_tag, updated_at, updated_by
		FROM notes
		WHERE patient_id = ?
		""",
		(patient_id,),
	).fetchone()
	if row is None:
		return {
			"note": "",
			"priority_tag": "None",
			"updated_at": "",
			"updated_by": "",
		}
	return dict(row)


def get_note_history(patient_id: int) -> pd.DataFrame:
	conn = get_db_connection()
	rows = conn.execute(
		"""
		SELECT note, priority_tag, created_at, created_by
		FROM note_history
		WHERE patient_id = ?
		ORDER BY created_at DESC
		LIMIT 50
		""",
		(patient_id,),
	).fetchall()
	return pd.DataFrame([dict(row) for row in rows])


def save_note(patient_id: int, note: str, priority_tag: str, username: str) -> None:
	conn = get_db_connection()
	stamp = datetime.utcnow().isoformat(timespec="seconds")
	conn.execute(
		"""
		INSERT INTO notes (patient_id, note, priority_tag, updated_at, updated_by)
		VALUES (?, ?, ?, ?, ?)
		ON CONFLICT(patient_id) DO UPDATE SET
			note = excluded.note,
			priority_tag = excluded.priority_tag,
			updated_at = excluded.updated_at,
			updated_by = excluded.updated_by
		""",
		(patient_id, note, priority_tag, stamp, username),
	)
	conn.execute(
		"""
		INSERT INTO note_history (patient_id, note, priority_tag, created_at, created_by)
		VALUES (?, ?, ?, ?, ?)
		""",
		(patient_id, note, priority_tag, stamp, username),
	)
	conn.commit()


def init_session_state() -> None:
	if "auth_user" not in st.session_state:
		st.session_state["auth_user"] = None
	if "auth_role" not in st.session_state:
		st.session_state["auth_role"] = None


def login_panel() -> bool:
	st.sidebar.subheader("Login")
	if st.session_state["auth_user"]:
		st.sidebar.success(
			f"Signed in as {st.session_state['auth_user']} "
			f"({st.session_state['auth_role']})"
		)
		if st.sidebar.button("Logout"):
			st.session_state["auth_user"] = None
			st.session_state["auth_role"] = None
			st.rerun()
		return True

	username = st.sidebar.text_input("Username")
	password = st.sidebar.text_input("Password", type="password")
	if st.sidebar.button("Sign in"):
		role = verify_user(username, password)
		if role:
			st.session_state["auth_user"] = username
			st.session_state["auth_role"] = role
			st.rerun()
		else:
			st.sidebar.error("Invalid username or password.")

	return st.session_state["auth_user"] is not None


def main() -> None:
	st.set_page_config(page_title="Patient Risk Triage", layout="wide")
	init_db()
	init_session_state()
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');

		:root {
			--ink: #10212b;
			--slate: #31404d;
			--accent: #0f9b88;
			--accent-2: #ff7a59;
			--accent-3: #5b7cfa;
			--surface: #f4f7fb;
			--surface-2: #e8eef6;
			--card: #ffffff;
			--border: #d6e2ee;
			--gold: #f3b33d;
		}

		html, body, [class*="css"] {
			font-family: 'Source Sans 3', sans-serif;
			color: var(--ink);
			background: var(--surface);
		}

		.main {
			background:
				radial-gradient(900px 520px at 8% -10%, rgba(91, 124, 250, 0.2), transparent 60%),
				radial-gradient(900px 520px at 90% 0%, rgba(255, 122, 89, 0.22), transparent 55%),
				radial-gradient(700px 420px at 85% 80%, rgba(15, 155, 136, 0.18), transparent 60%),
				linear-gradient(180deg, #f4f7fb 0%, #ffffff 40%, #f7f1ec 100%);
		}

		.block-container {
			padding-top: 1.5rem;
			padding-bottom: 2.5rem;
		}

		.app-shell {
			background: rgba(255, 255, 255, 0.92);
			backdrop-filter: blur(6px);
			border-radius: 22px;
			padding: 1.8rem 2rem 2rem;
			border: 1px solid var(--border);
			box-shadow: 0 20px 50px rgba(16, 33, 43, 0.14);
		}

		.hero {
			display: grid;
			grid-template-columns: minmax(0, 1.3fr) minmax(0, 0.9fr);
			gap: 1.5rem;
			align-items: center;
			padding: 1.4rem 1.6rem;
			border-radius: 18px;
			background: linear-gradient(125deg, #0f3d33, #0f9b88 45%, #5b7cfa 100%);
			color: #f7fbfd;
			margin-bottom: 1.4rem;
		}

		.hero-badge {
			display: inline-flex;
			align-items: center;
			gap: 0.45rem;
			font-size: 0.8rem;
			font-weight: 700;
			letter-spacing: 0.08em;
			text-transform: uppercase;
			background: rgba(255, 255, 255, 0.2);
			border: 1px solid rgba(255, 255, 255, 0.3);
			padding: 0.25rem 0.7rem;
			border-radius: 999px;
		}

		.hero-title {
			font-family: 'Playfair Display', serif;
			font-size: 2.2rem;
			font-weight: 700;
			margin: 0;
		}

		.hero-subtitle {
			margin: 0.35rem 0 0;
			color: #cfe6ee;
			font-size: 1rem;
		}

		.hero-cta {
			display: flex;
			gap: 0.75rem;
			margin-top: 1rem;
		}

		.cta-primary {
			background: #ffffff;
			color: #0f3d33;
			border-radius: 999px;
			padding: 0.5rem 1.1rem;
			font-weight: 700;
			border: none;
		}

		.cta-secondary {
			background: rgba(255, 255, 255, 0.18);
			color: #ffffff;
			border-radius: 999px;
			padding: 0.5rem 1.1rem;
			font-weight: 600;
			border: 1px solid rgba(255, 255, 255, 0.4);
		}

		.hero-panel {
			background: rgba(255, 255, 255, 0.16);
			border: 1px solid rgba(255, 255, 255, 0.3);
			border-radius: 16px;
			padding: 1rem 1.1rem;
			color: #eaf3f6;
		}

		.hero-stats {
			display: grid;
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: 0.8rem;
			margin-top: 0.75rem;
		}

		.hero-stat {
			background: rgba(255, 255, 255, 0.18);
			border-radius: 12px;
			padding: 0.6rem 0.75rem;
			border: 1px solid rgba(255, 255, 255, 0.25);
		}

		.hero-stat span {
			font-size: 0.7rem;
			letter-spacing: 0.08em;
			text-transform: uppercase;
			color: rgba(255, 255, 255, 0.7);
		}

		.hero-stat strong {
			display: block;
			font-size: 1.1rem;
			color: #ffffff;
			margin-top: 0.25rem;
		}

		.section-title {
			font-family: 'Playfair Display', serif;
			font-size: 1.35rem;
			margin-bottom: 0.4rem;
			color: var(--ink);
		}

		.section-caption {
			color: #2f3a45;
			font-weight: 500;
			margin-top: 0;
		}

		.stTabs [data-baseweb="tab"] {
			font-size: 0.95rem;
			font-weight: 600;
			padding: 0.6rem 1.1rem;
			color: #ffffff;
			background: linear-gradient(120deg, rgba(15, 155, 136, 0.7), rgba(91, 124, 250, 0.7));
			border-radius: 999px;
		}

		.stTabs [aria-selected="true"] {
			background: linear-gradient(120deg, rgba(15, 155, 136, 0.95), rgba(91, 124, 250, 0.95));
			color: #ffffff;
			border-radius: 999px;
		}

		.stButton > button {
			background: linear-gradient(120deg, #0f9b88, #5b7cfa);
			color: white;
			border-radius: 10px;
			border: none;
			padding: 0.55rem 1.1rem;
			font-weight: 600;
		}

		.stButton > button:hover {
			background: linear-gradient(120deg, #0b6f62, #3f5edd);
		}

		.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
			border-radius: 8px;
			border-color: var(--border);
		}

		div[data-testid="stDataFrame"] {
			border-radius: 12px;
			border: 1px solid var(--border);
		}

		.section-card {
			background: var(--card);
			border: 1px solid var(--border);
			border-radius: 16px;
			padding: 1.1rem 1.2rem 1.25rem;
			box-shadow: 0 14px 30px rgba(20, 33, 43, 0.08);
			margin-bottom: 1.2rem;
		}

		.section-card h2, .section-card h3 {
			margin-top: 0;
		}

		[data-testid="stSidebar"] {
			background: #eaf4ff;
			color: var(--ink);
			border-right: 1px solid var(--border);
		}

		[data-testid="stSidebar"] h2,
		[data-testid="stSidebar"] label,
		[data-testid="stSidebar"] p {
			color: var(--ink);
		}

		[data-testid="stSidebar"] .stButton > button {
			background: linear-gradient(120deg, #0f9b88, #5b7cfa);
			color: #ffffff;
		}

		[data-testid="stSidebar"] input {
			background: #ffffff;
			color: #0b1b24;
		}

		.image-grid {
			display: grid;
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: 1rem;
			margin-top: 1rem;
			margin-bottom: 1rem;
		}

		.image-card {
			display: grid;
			grid-template-columns: 120px minmax(0, 1fr);
			gap: 1rem;
			align-items: center;
			background: var(--card);
			border-radius: 16px;
			border: 1px solid var(--border);
			padding: 0.8rem 1rem;
			box-shadow: 0 12px 25px rgba(16, 33, 43, 0.1);
		}

		.image-card img {
			width: 120px;
			height: auto;
		}

		.image-card h5 {
			margin: 0 0 0.4rem;
			font-size: 1rem;
			color: var(--ink);
		}

		.image-card p {
			margin: 0;
			color: var(--slate);
			font-size: 0.9rem;
		}

		.feature-grid {
			display: grid;
			grid-template-columns: repeat(3, minmax(0, 1fr));
			gap: 1rem;
			margin: 1.1rem 0 0.2rem;
		}

		.feature-card {
			background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(240, 247, 255, 0.7));
			border: 1px solid var(--border);
			border-radius: 16px;
			padding: 1rem 1.1rem;
			box-shadow: 0 12px 25px rgba(15, 42, 67, 0.08);
		}

		.feature-card h4 {
			margin: 0 0 0.35rem;
			font-size: 1rem;
			color: var(--ink);
		}

		.feature-card p {
			margin: 0;
			color: var(--slate);
			font-size: 0.92rem;
		}

		.login-note {
			background: #e7f1f5;
			border-left: 4px solid var(--accent);
			padding: 0.8rem 1rem;
			border-radius: 10px;
			color: var(--slate);
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
	if not login_panel():
		st.markdown(
			"""
			<div class="login-note">
				<strong>Secure access required.</strong> Please sign in to access the dashboard.
			</div>
			""",
			unsafe_allow_html=True,
		)
		return

	st.markdown(
		"""
		<div class="app-shell">
			<div class="hero">
				<div>
					<div class="hero-badge">Medicare platform</div>
					<h1 class="hero-title">Medicare</h1>
					<p class="hero-subtitle">A colorful, intuitive command center for maternal and diabetes care teams. Predict risk, rank priorities, and coordinate follow ups with confidence.</p>
					<div class="hero-cta">
						<button class="cta-primary">Start with a CSV upload</button>
						<button class="cta-secondary">View risk overview</button>
					</div>
				</div>
				<div class="hero-panel">
					<strong>Operational pulse</strong>
					<p style="margin: 0.35rem 0 0.6rem; color: rgba(255, 255, 255, 0.75);">
						Unified visibility across care lines.
					</p>
					<div class="hero-stats">
						<div class="hero-stat">
							<span>Coverage</span>
							<strong>Maternity + Diabetes</strong>
						</div>
						<div class="hero-stat">
							<span>Priority</span>
							<strong>Risk Ranking</strong>
						</div>
						<div class="hero-stat">
							<span>Intake</span>
							<strong>CSV Ready</strong>
						</div>
					</div>
				</div>
			</div>
			<div class="feature-grid">
				<div class="feature-card">
					<h4>Color coded triage</h4>
					<p>See priority lists, clinician notes, and risk scores in one view.</p>
				</div>
				<div class="feature-card">
					<h4>Fast clinical review</h4>
					<p>Focus on the riskiest patients with sortable, ranked insights.</p>
				</div>
				<div class="feature-card">
					<h4>Confident operations</h4>
					<p>Standardize assessments across maternity and diabetes pathways.</p>
				</div>
			</div>
			<h2 class="section-title">Patient Risk Triage and Prediction</h2>
			<p class="section-caption">Upload patient data to train a model and prioritize patients by predicted risk. The model is for decision support and does not replace clinical judgment.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	render_medical_gallery()

	tabs = st.tabs(["Maternity", "Diabetes"])

	with tabs[0]:
		render_maternity_tab()

	with tabs[1]:
		render_diabetes_tab()


def render_maternity_tab() -> None:
	st.header("Maternity Patients")
	uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="maternity")

	try:
		data, report = prepare_data(uploaded_file)
	except Exception as exc:
		st.error(f"Data load failed: {exc}")
		return

	data = data.reset_index(drop=True)
	data.insert(0, "Patient ID", data.index + 1)

	st.markdown('<div class="section-card">', unsafe_allow_html=True)
	st.subheader("Data Snapshot")
	st.dataframe(data.head(20), use_container_width=True)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown('<div class="section-card">', unsafe_allow_html=True)
	st.subheader("Model Training")
	X = data[FEATURE_COLUMNS]
	y = data[RISK_COLUMN]

	if y.nunique() < 2:
		st.warning("Need at least two risk classes to train a model.")
		return

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	model = build_model()
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	st.text("Classification report:")
	st.text(classification_report(y_test, y_pred))

	try:
		proba = model.predict_proba(X_test)
		class_labels = list(model.named_steps["classifier"].classes_)
		if "High" in class_labels:
			high_index = class_labels.index("High")
			auc = roc_auc_score((y_test == "High").astype(int), proba[:, high_index])
			st.write(f"ROC AUC (High vs Rest): {auc:.3f}")
	except Exception:
		pass

	st.text("Confusion matrix:")
	st.write(confusion_matrix(y_test, y_pred))
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown('<div class="section-card">', unsafe_allow_html=True)
	st.subheader("Patient Priority List")
	priority_scores = get_priority_scores(model, X)
	priority_table = data.copy()
	priority_table["Risk Score"] = priority_scores
	priority_table = priority_table.sort_values(by="Risk Score", ascending=False)
	notes_map = get_notes_map(priority_table["Patient ID"].tolist())
	priority_table["Notes"] = priority_table["Patient ID"].map(notes_map).fillna("")
	tags_map = get_tags_map(priority_table["Patient ID"].tolist())
	priority_table["Priority Tag"] = priority_table["Patient ID"].map(tags_map).fillna("None")
	st.dataframe(priority_table.head(30), use_container_width=True)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown('<div class="section-card">', unsafe_allow_html=True)
	st.subheader("Patient Notes")
	selected_id = st.selectbox(
		"Select patient ID",
		priority_table["Patient ID"].tolist(),
		key="maternity_notes",
	)
	selected_row = priority_table.loc[priority_table["Patient ID"] == selected_id].iloc[0]
	st.write(
		f"Selected patient - Age: {selected_row['Age']}, "
		f"BP: {selected_row['Systolic BP']}/{selected_row['Diastolic']}, "
		f"Risk: {selected_row[RISK_COLUMN]}"
	)
	latest_record = get_latest_note_record(selected_id)
	st.text_area("Latest note", value=latest_record["note"], height=120, disabled=True)
	st.write(f"Latest priority tag: {latest_record['priority_tag']}")
	if latest_record["updated_at"]:
		st.write(
			f"Last updated: {latest_record['updated_at']} by {latest_record['updated_by']}"
		)

	if st.session_state["auth_role"] == "doctor":
		selected_tag = st.selectbox(
			"Priority tag",
			PRIORITY_TAGS,
			key="maternity_tag",
		)
		new_note = st.text_area(
			"Add note",
			value="",
			height=120,
			key="maternity_note",
		)
		if st.button("Save note", key="maternity_save"):
			if new_note.strip():
				save_note(
					selected_id,
					new_note.strip(),
					selected_tag,
					st.session_state["auth_user"],
				)
				st.success("Note saved.")
			else:
				st.warning("Note is empty.")

	note_history = get_note_history(selected_id)
	st.write("Note history (latest first):")
	if note_history.empty:
		st.info("No notes yet for this patient.")
	else:
		st.dataframe(note_history, use_container_width=True)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown('<div class="section-card">', unsafe_allow_html=True)
	st.subheader("Single Patient Prediction")
	default_values = X.median(numeric_only=True).to_dict()
	with st.form("patient_form"):
		form_values = {}
		for feature in FEATURE_COLUMNS:
			form_values[feature] = st.number_input(
				feature,
				value=float(default_values.get(feature, 0.0)),
			)
		submitted = st.form_submit_button("Predict")

	if submitted:
		patient_df = pd.DataFrame([form_values])
		prediction = model.predict(patient_df)[0]
		risk_score = float(get_priority_scores(model, patient_df)[0])
		st.write(f"Predicted risk: {prediction}")
		st.write(f"Risk score: {risk_score:.3f}")

		summary = build_patient_summary(patient_df.iloc[0], X_train)
		st.write("Key patient signals:")
		for item in summary:
			st.write(f"- {item}")
	st.markdown("</div>", unsafe_allow_html=True)


def render_diabetes_tab() -> None:
	st.header("Diabetes Patients")
	uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="diabetes")

	try:
		data, report = prepare_diabetes_data(uploaded_file)
	except Exception as exc:
		st.error(f"Data load failed: {exc}")
		return

	data = data.reset_index(drop=True)
	data.insert(0, "Patient ID", data.index + 1)

	available_categoricals = [
		col for col in DIABETES_CATEGORICAL_COLUMNS if col in data.columns
	]
	feature_columns = [
		col
		for col in data.columns
		if col not in {"Patient ID", DIABETES_TARGET, *DIABETES_EXCLUDE_COLUMNS}
	]
	numeric_columns = [
		col for col in feature_columns if col not in available_categoricals
	]
	X = data[feature_columns]
	y = data[DIABETES_TARGET].astype(int)

	st.markdown('<div class="section-card">', unsafe_allow_html=True)
	st.subheader("Data Snapshot")
	st.dataframe(data.head(20), use_container_width=True)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown('<div class="section-card">', unsafe_allow_html=True)
	st.subheader("Model Training")
	if y.nunique() < 2:
		st.warning("Need both 0 and 1 classes to train a model.")
		return

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	model = build_diabetes_model(numeric_columns, available_categoricals)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	st.text("Classification report:")
	st.text(classification_report(y_test, y_pred))

	try:
		proba = model.predict_proba(X_test)
		class_labels = list(model.named_steps["classifier"].classes_)
		if 1 in class_labels:
			pos_index = class_labels.index(1)
			auc = roc_auc_score(y_test, proba[:, pos_index])
			st.write(f"ROC AUC (Diabetes vs Rest): {auc:.3f}")
	except Exception:
		pass

	st.text("Confusion matrix:")
	st.write(confusion_matrix(y_test, y_pred))
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown('<div class="section-card">', unsafe_allow_html=True)
	st.subheader("Patient Priority List")
	threshold = st.slider(
		"High priority threshold",
		min_value=0.1,
		max_value=0.9,
		value=0.5,
		step=0.05,
	)
	priority_scores = get_binary_risk_scores(model, X)
	priority_table = data.copy()
	priority_table["Risk Score"] = priority_scores
	priority_table["Priority"] = np.where(
		priority_table["Risk Score"] >= threshold, "High", "Low"
	)
	priority_table = priority_table.sort_values(by="Risk Score", ascending=False)
	st.dataframe(priority_table.head(30), use_container_width=True)
	st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
	main()
