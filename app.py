from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Student Progress Dashboard",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "preprocess": DATA_DIR / "preprocess_students.xlsx",
    "registered": DATA_DIR / "registered_students.xlsx",
    "graduated": DATA_DIR / "graduated_students.xlsx",
    "supervisors": DATA_DIR / "supervisor_student_report.xlsx",
    "examiners": DATA_DIR / "external_examiner_report.xlsx",
}


# =========================================================
# HELPERS
# =========================================================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace({np.nan: None})
            df[col] = df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
    return df


def extract_degree_level(programme_value):
    if pd.isna(programme_value):
        return "Unknown"

    text = str(programme_value).lower()

    if "doctor of philosophy" in text or "phd" in text:
        return "PhD"
    if "master of science" in text:
        return "MSc"
    if "master of agriculture" in text:
        return "M Agric"

    return "Other"


def clean_completion_year(series: pd.Series) -> pd.Series:
    """
    Convert completion year to numeric and replace 2000 with NA.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.mask(s == 2000, pd.NA)
    return s


def clean_completion_date_or_year(series: pd.Series) -> pd.Series:
    """
    Some of your exports seem to store just the year in Completion Date.
    This converts values to numeric where possible and treats 2000 as NA.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.mask(s == 2000, pd.NA)
    return s


def normalize_role(role_value):
    if pd.isna(role_value):
        return "Unknown"

    role = str(role_value).strip().upper()

    if role in {"PRIMARY", "MAIN", "LEAD"}:
        return "Primary"
    if role in {"SECONDARY", "CO", "CO-SUPERVISOR", "COSUPERVISOR"}:
        return "Co-supervisor"

    return str(role_value).title()


def find_first_matching_column(df: pd.DataFrame, candidates: list[str]):
    cols_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def add_programme_and_degree(df: pd.DataFrame, programme_col: str = "Programme") -> pd.DataFrame:
    df = df.copy()
    if programme_col in df.columns:
        df["Degree Level"] = df[programme_col].apply(extract_degree_level)
    else:
        df["Degree Level"] = "Unknown"
    return df


def safe_value_counts(df: pd.DataFrame, col: str, dropna_label: str = "Unknown") -> pd.DataFrame:
    temp = df.copy()
    temp[col] = temp[col].fillna(dropna_label)
    out = temp[col].value_counts().reset_index()
    out.columns = [col, "Count"]
    return out


def filter_dataframe(df: pd.DataFrame, programme_col="Programme", degree_col="Degree Level"):
    filtered = df.copy()

    if programme_col in filtered.columns:
        programme_options = sorted([x for x in filtered[programme_col].dropna().unique()])
        selected_programmes = st.sidebar.multiselect(
            "Programme",
            options=programme_options,
            default=[]
        )
        if selected_programmes:
            filtered = filtered[filtered[programme_col].isin(selected_programmes)]

    if degree_col in filtered.columns:
        degree_options = sorted([x for x in filtered[degree_col].dropna().unique()])
        selected_degrees = st.sidebar.multiselect(
            "Degree Level",
            options=degree_options,
            default=[]
        )
        if selected_degrees:
            filtered = filtered[filtered[degree_col].isin(selected_degrees)]

    return filtered


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_all_data():
    missing = [name for name, path in FILES.items() if not path.exists()]
    if missing:
        return None, missing

    preprocess = pd.read_excel(FILES["preprocess"])
    registered = pd.read_excel(FILES["registered"])
    graduated = pd.read_excel(FILES["graduated"])

    supervisor_summary = pd.read_excel(FILES["supervisors"], sheet_name="Supervisor Summary")
    supervisor_students = pd.read_excel(FILES["supervisors"], sheet_name="Supervisor Students")

    examiner_summary = pd.read_excel(FILES["examiners"], sheet_name="Examiner Summary")
    examiner_detail = pd.read_excel(FILES["examiners"], sheet_name="Examiner Detail")

    return {
        "preprocess": preprocess,
        "registered": registered,
        "graduated": graduated,
        "supervisor_summary": supervisor_summary,
        "supervisor_students": supervisor_students,
        "examiner_summary": examiner_summary,
        "examiner_detail": examiner_detail,
    }, None


def prepare_dataframes(data_dict):
    preprocess = clean_text_columns(standardize_columns(data_dict["preprocess"]))
    registered = clean_text_columns(standardize_columns(data_dict["registered"]))
    graduated = clean_text_columns(standardize_columns(data_dict["graduated"]))
    supervisor_summary = clean_text_columns(standardize_columns(data_dict["supervisor_summary"]))
    supervisor_students = clean_text_columns(standardize_columns(data_dict["supervisor_students"]))
    examiner_summary = clean_text_columns(standardize_columns(data_dict["examiner_summary"]))
    examiner_detail = clean_text_columns(standardize_columns(data_dict["examiner_detail"]))

    # Add degree level
    preprocess = add_programme_and_degree(preprocess)
    registered = add_programme_and_degree(registered)
    graduated = add_programme_and_degree(graduated)
    supervisor_students = add_programme_and_degree(supervisor_students)
    examiner_detail = add_programme_and_degree(examiner_detail)

    # Clean completion year/date in graduated
    if "Completion Year" in graduated.columns:
        graduated["Completion Year"] = clean_completion_year(graduated["Completion Year"])

    if "Completion Date" in graduated.columns:
        graduated["Completion Date"] = clean_completion_date_or_year(graduated["Completion Date"])

    # Clean completion year/date in supervisor detail
    if "Completion Year" in supervisor_students.columns:
        supervisor_students["Completion Year"] = clean_completion_year(supervisor_students["Completion Year"])

    if "Completion Date" in supervisor_students.columns:
        supervisor_students["Completion Date"] = clean_completion_date_or_year(supervisor_students["Completion Date"])

    # Clean completion year/date in examiner detail
    if "Completion Year" in examiner_detail.columns:
        examiner_detail["Completion Year"] = clean_completion_year(examiner_detail["Completion Year"])

    if "Completion Date" in examiner_detail.columns:
        examiner_detail["Completion Date"] = clean_completion_date_or_year(examiner_detail["Completion Date"])

    # Normalize roles
    if "Role" in supervisor_students.columns:
        supervisor_students["Normalized Role"] = supervisor_students["Role"].apply(normalize_role)

    if "Examiner Role" in examiner_detail.columns:
        examiner_detail["Normalized Examiner Role"] = examiner_detail["Examiner Role"].apply(normalize_role)

    return {
        "preprocess": preprocess,
        "registered": registered,
        "graduated": graduated,
        "supervisor_summary": supervisor_summary,
        "supervisor_students": supervisor_students,
        "examiner_summary": examiner_summary,
        "examiner_detail": examiner_detail,
    }


# =========================================================
# MAIN LOAD
# =========================================================
data_raw, missing_files = load_all_data()

if missing_files:
    st.error("Some required Excel files are missing from the local data folder.")
    st.write("Expected files:")
    for name, path in FILES.items():
        st.write(f"- {path.name}")
    st.stop()

data = prepare_dataframes(data_raw)

preprocess = data["preprocess"]
registered = data["registered"]
graduated = data["graduated"]
supervisor_summary = data["supervisor_summary"]
supervisor_students = data["supervisor_students"]
examiner_summary = data["examiner_summary"]
examiner_detail = data["examiner_detail"]


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Filters")

all_programmes = sorted(
    set(preprocess.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(registered.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(graduated.get("Programme", pd.Series(dtype=object)).dropna().tolist())
)

all_degrees = sorted(
    set(preprocess.get("Degree Level", pd.Series(dtype=object)).dropna().tolist())
    | set(registered.get("Degree Level", pd.Series(dtype=object)).dropna().tolist())
    | set(graduated.get("Degree Level", pd.Series(dtype=object)).dropna().tolist())
)

selected_programmes = st.sidebar.multiselect("Programme", options=all_programmes, default=[])
selected_degrees = st.sidebar.multiselect("Degree Level", options=all_degrees, default=[])

graduation_year_options = []
if "Completion Year" in graduated.columns:
    graduation_year_options = sorted(
        [int(x) for x in graduated["Completion Year"].dropna().unique()]
    )

selected_grad_years = st.sidebar.multiselect(
    "Graduation Year",
    options=graduation_year_options,
    default=[]
)


def apply_main_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if selected_programmes and "Programme" in out.columns:
        out = out[out["Programme"].isin(selected_programmes)]

    if selected_degrees and "Degree Level" in out.columns:
        out = out[out["Degree Level"].isin(selected_degrees)]

    if selected_grad_years and "Completion Year" in out.columns:
        out = out[out["Completion Year"].isin(selected_grad_years)]

    return out


preprocess_f = apply_main_filters(preprocess)
registered_f = apply_main_filters(registered)
graduated_f = apply_main_filters(graduated)
supervisor_students_f = apply_main_filters(supervisor_students)
examiner_detail_f = apply_main_filters(examiner_detail)


# =========================================================
# TITLE
# =========================================================
st.title("Postgraduate Student Dashboard")
st.caption("Data loaded from local Excel exports in the data folder.")


# =========================================================
# OVERVIEW
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Overview",
        "Pre-process",
        "Registered",
        "Graduated",
        "Supervisors",
        "External Examiners",
    ]
)

with tab1:
    st.subheader("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-process Students", len(preprocess_f))
    c2.metric("Registered Students", len(registered_f))
    c3.metric("Graduated Students", len(graduated_f))

    left, right = st.columns(2)

    with left:
        if "Workflow Decision Status" in preprocess_f.columns:
            pp_status = safe_value_counts(preprocess_f, "Workflow Decision Status", "Unknown")
            fig = px.bar(
                pp_status,
                x="Workflow Decision Status",
                y="Count",
                title="Pre-process Decision Status"
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if "Completion Year" in graduated_f.columns:
            grad_year = (
                graduated_f["Completion Year"]
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .reset_index()
            )
            if not grad_year.empty:
                grad_year.columns = ["Completion Year", "Count"]
                fig = px.bar(
                    grad_year,
                    x="Completion Year",
                    y="Count",
                    title="Graduations by Year"
                )
                st.plotly_chart(fig, use_container_width=True)

    lower_left, lower_right = st.columns(2)

    with lower_left:
        if "Degree Level" in registered_f.columns:
            reg_degree = safe_value_counts(registered_f, "Degree Level")
            fig = px.bar(
                reg_degree,
                x="Degree Level",
                y="Count",
                title="Registered Students by Degree"
            )
            st.plotly_chart(fig, use_container_width=True)

    with lower_right:
        if "Degree Level" in graduated_f.columns:
            grad_degree = safe_value_counts(graduated_f, "Degree Level")
            fig = px.bar(
                grad_degree,
                x="Degree Level",
                y="Count",
                title="Graduated Students by Degree"
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PRE-PROCESS
# =========================================================
with tab2:
    st.subheader("Pre-process Students")

    if "Assigned Supervisor" in preprocess_f.columns:
        left, right = st.columns(2)

        with left:
            if "Workflow Decision Status" in preprocess_f.columns:
                status_summary = safe_value_counts(preprocess_f, "Workflow Decision Status", "Unknown")
                fig = px.bar(
                    status_summary,
                    x="Workflow Decision Status",
                    y="Count",
                    title="Workflow Decision Status"
                )
                st.plotly_chart(fig, use_container_width=True)

        with right:
            supervisor_summary_pp = safe_value_counts(preprocess_f, "Assigned Supervisor", "Unassigned")
            fig = px.bar(
                supervisor_summary_pp.head(15),
                x="Assigned Supervisor",
                y="Count",
                title="Assigned Supervisors (Top 15)"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Student Table")
    st.dataframe(preprocess_f, use_container_width=True)


# =========================================================
# REGISTERED
# =========================================================
with tab3:
    st.subheader("Registered Students")

    col1, col2 = st.columns(2)

    with col1:
        if "Degree Level" in registered_f.columns:
            reg_degree = safe_value_counts(registered_f, "Degree Level")
            fig = px.bar(
                reg_degree,
                x="Degree Level",
                y="Count",
                title="Registered Students by Degree"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        workflow_col = find_first_matching_column(registered_f, ["Workflow State", "Workflow Status"])
        if workflow_col:
            reg_workflow = safe_value_counts(registered_f, workflow_col, "Unknown")
            fig = px.bar(
                reg_workflow,
                x=workflow_col,
                y="Count",
                title="Registered Students by Workflow State"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Student Table")
    st.dataframe(registered_f, use_container_width=True)


# =========================================================
# GRADUATED
# =========================================================
with tab4:
    st.subheader("Graduated Students")

    col1, col2 = st.columns(2)

    with col1:
        if "Completion Year" in graduated_f.columns:
            grad_year = (
                graduated_f["Completion Year"]
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .reset_index()
            )
            if not grad_year.empty:
                grad_year.columns = ["Completion Year", "Count"]
                fig = px.bar(
                    grad_year,
                    x="Completion Year",
                    y="Count",
                    title="Graduated Students by Year"
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Degree Level" in graduated_f.columns:
            grad_degree = safe_value_counts(graduated_f, "Degree Level")
            fig = px.bar(
                grad_degree,
                x="Degree Level",
                y="Count",
                title="Graduated Students by Degree"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Student Table")
    st.dataframe(graduated_f, use_container_width=True)


# =========================================================
# SUPERVISORS
# =========================================================
with tab5:
    st.subheader("Supervisor Dashboard")

    sup_names = sorted(
        [x for x in supervisor_students_f["Supervisor"].dropna().unique()]
    ) if "Supervisor" in supervisor_students_f.columns else []

    selected_supervisors = st.multiselect(
        "Select supervisor(s)",
        options=sup_names,
        default=[]
    )

    sup_df = supervisor_students_f.copy()
    if selected_supervisors and "Supervisor" in sup_df.columns:
        sup_df = sup_df[sup_df["Supervisor"].isin(selected_supervisors)]

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        if "Supervisor" in sup_df.columns:
            sup_counts = safe_value_counts(sup_df, "Supervisor", "Unknown")
            fig = px.bar(
                sup_counts.head(20),
                x="Supervisor",
                y="Count",
                title="Students per Supervisor"
            )
            st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        if "Normalized Role" in sup_df.columns:
            role_counts = safe_value_counts(sup_df, "Normalized Role", "Unknown")
            fig = px.bar(
                role_counts,
                x="Normalized Role",
                y="Count",
                title="Supervisor Role Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        if {"Supervisor", "Student Stage"}.issubset(sup_df.columns):
            stage_by_sup = (
                sup_df.groupby(["Supervisor", "Student Stage"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                stage_by_sup,
                x="Supervisor",
                y="Count",
                color="Student Stage",
                barmode="group",
                title="Registered vs Graduated by Supervisor"
            )
            st.plotly_chart(fig, use_container_width=True)

    with row2_col2:
        if {"Supervisor", "Degree Level"}.issubset(sup_df.columns):
            degree_by_sup = (
                sup_df.groupby(["Supervisor", "Degree Level"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                degree_by_sup,
                x="Supervisor",
                y="Count",
                color="Degree Level",
                barmode="stack",
                title="Degree Allocation by Supervisor"
            )
            st.plotly_chart(fig, use_container_width=True)

    if {"Supervisor", "Completion Year"}.issubset(sup_df.columns):
        sup_grad = sup_df[sup_df["Student Stage"].astype(str).str.lower() == "graduated"].copy()
        sup_grad = sup_grad.dropna(subset=["Completion Year"])

        if not sup_grad.empty:
            sup_grad["Completion Year"] = sup_grad["Completion Year"].astype(int)
            yearly_sup = (
                sup_grad.groupby(["Completion Year", "Supervisor"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                yearly_sup,
                x="Completion Year",
                y="Count",
                color="Supervisor",
                title="Graduated Students by Year and Supervisor"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Supervisor Detail")
    st.dataframe(sup_df, use_container_width=True)


# =========================================================
# EXTERNAL EXAMINERS
# =========================================================
with tab6:
    st.subheader("External Examiner Dashboard")

    examiner_col = "External Examiner" if "External Examiner" in examiner_detail_f.columns else None

    examiner_names = (
        sorted([x for x in examiner_detail_f[examiner_col].dropna().unique()])
        if examiner_col
        else []
    )

    selected_examiners = st.multiselect(
        "Select external examiner(s)",
        options=examiner_names,
        default=[]
    )

    ex_df = examiner_detail_f.copy()
    if selected_examiners and examiner_col:
        ex_df = ex_df[ex_df[examiner_col].isin(selected_examiners)]

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        if examiner_col:
            ex_counts = safe_value_counts(ex_df, examiner_col, "Unknown")
            fig = px.bar(
                ex_counts.head(20),
                x=examiner_col,
                y="Count",
                title="Students per External Examiner"
            )
            st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        if "Normalized Examiner Role" in ex_df.columns:
            ex_role_counts = safe_value_counts(ex_df, "Normalized Examiner Role", "Unknown")
            fig = px.bar(
                ex_role_counts,
                x="Normalized Examiner Role",
                y="Count",
                title="Examiner Role Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        if examiner_col and "Student Stage" in ex_df.columns:
            stage_by_examiner = (
                ex_df.groupby([examiner_col, "Student Stage"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                stage_by_examiner,
                x=examiner_col,
                y="Count",
                color="Student Stage",
                barmode="group",
                title="Registered vs Graduated by External Examiner"
            )
            st.plotly_chart(fig, use_container_width=True)

    with row2_col2:
        if examiner_col and "Degree Level" in ex_df.columns:
            degree_by_examiner = (
                ex_df.groupby([examiner_col, "Degree Level"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                degree_by_examiner,
                x=examiner_col,
                y="Count",
                color="Degree Level",
                barmode="stack",
                title="Degree Allocation by External Examiner"
            )
            st.plotly_chart(fig, use_container_width=True)

    if examiner_col and "Completion Year" in ex_df.columns:
        ex_grad = ex_df[ex_df["Student Stage"].astype(str).str.lower() == "graduated"].copy()
        ex_grad = ex_grad.dropna(subset=["Completion Year"])

        if not ex_grad.empty:
            ex_grad["Completion Year"] = ex_grad["Completion Year"].astype(int)
            yearly_examiner = (
                ex_grad.groupby(["Completion Year", examiner_col])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                yearly_examiner,
                x="Completion Year",
                y="Count",
                color=examiner_col,
                title="Graduated Students by Year and External Examiner"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### External Examiner Detail")
    st.dataframe(ex_df, use_container_width=True)
