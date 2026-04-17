import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Student Progress Dashboard", layout="wide")


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
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def extract_degree_level(programme_value):
    if pd.isna(programme_value):
        return "Unknown"

    text = str(programme_value).lower()

    if "doctor of philosophy" in text or "phd" in text:
        return "PhD"
    if "master of science" in text:
        return "MSc"
    if "master of agriculture" in text or "magric" in text:
        return "M Agric"
    if "masters" in text or "master" in text:
        return "Master's"
    return "Other"


def clean_completion_year(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.mask(s == 2000, pd.NA)
    return s


def clean_completion_date_or_year(series: pd.Series) -> pd.Series:
    parsed_dates = pd.to_datetime(series, errors="coerce")
    if parsed_dates.notna().sum() > 0:
        years = parsed_dates.dt.year
        years = years.mask(years == 2000, pd.NA)
        return years

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


def prepare_dataframes(data_dict):
    preprocess = clean_text_columns(standardize_columns(data_dict["preprocess"]))
    registered = clean_text_columns(standardize_columns(data_dict["registered"]))
    graduated = clean_text_columns(standardize_columns(data_dict["graduated"]))
    supervisor_students = clean_text_columns(standardize_columns(data_dict["supervisor_students"]))
    examiner_detail = clean_text_columns(standardize_columns(data_dict["examiner_detail"]))

    preprocess = add_programme_and_degree(preprocess)
    registered = add_programme_and_degree(registered)
    graduated = add_programme_and_degree(graduated)
    supervisor_students = add_programme_and_degree(supervisor_students)
    examiner_detail = add_programme_and_degree(examiner_detail)

    if "Completion Year" in graduated.columns:
        graduated["Completion Year"] = clean_completion_year(graduated["Completion Year"])
    elif "Completion Date" in graduated.columns:
        graduated["Completion Year"] = clean_completion_date_or_year(graduated["Completion Date"])

    if "Completion Year" in supervisor_students.columns:
        supervisor_students["Completion Year"] = clean_completion_year(supervisor_students["Completion Year"])
    elif "Completion Date" in supervisor_students.columns:
        supervisor_students["Completion Year"] = clean_completion_date_or_year(supervisor_students["Completion Date"])

    if "Completion Year" in examiner_detail.columns:
        examiner_detail["Completion Year"] = clean_completion_year(examiner_detail["Completion Year"])
    elif "Completion Date" in examiner_detail.columns:
        examiner_detail["Completion Year"] = clean_completion_date_or_year(examiner_detail["Completion Date"])

    if "Role" in supervisor_students.columns:
        supervisor_students["Normalized Role"] = supervisor_students["Role"].apply(normalize_role)

    if "Examiner Role" in examiner_detail.columns:
        examiner_detail["Normalized Examiner Role"] = examiner_detail["Examiner Role"].apply(normalize_role)

    return preprocess, registered, graduated, supervisor_students, examiner_detail


@st.cache_data
def load_all_data(preprocess_file, registered_file, graduated_file, supervisor_file, examiner_file):
    preprocess = pd.read_excel(preprocess_file)
    registered = pd.read_excel(registered_file)
    graduated = pd.read_excel(graduated_file)
    supervisor_students = pd.read_excel(supervisor_file, sheet_name="Supervisor Students")
    examiner_detail = pd.read_excel(examiner_file, sheet_name="Examiner Detail")

    return {
        "preprocess": preprocess,
        "registered": registered,
        "graduated": graduated,
        "supervisor_students": supervisor_students,
        "examiner_detail": examiner_detail,
    }


# =========================================================
# TITLE
# =========================================================
st.title("Postgraduate Student Dashboard")
st.write("Upload or drag and drop the five Excel files below.")


# =========================================================
# FILE UPLOAD AREA
# =========================================================
col1, col2 = st.columns(2)

with col1:
    preprocess_file = st.file_uploader(
        "Pre-process students",
        type="xlsx",
        key="preprocess_file",
        help="Upload preprocess_students.xlsx"
    )
    registered_file = st.file_uploader(
        "Registered students",
        type="xlsx",
        key="registered_file",
        help="Upload registered_students.xlsx"
    )
    graduated_file = st.file_uploader(
        "Graduated students",
        type="xlsx",
        key="graduated_file",
        help="Upload graduated_students.xlsx"
    )

with col2:
    supervisor_file = st.file_uploader(
        "Supervisor student report",
        type="xlsx",
        key="supervisor_file",
        help="Upload supervisor_student_report.xlsx"
    )
    examiner_file = st.file_uploader(
        "External examiner report",
        type="xlsx",
        key="examiner_file",
        help="Upload external_examiner_report.xlsx"
    )


all_files_uploaded = all([
    preprocess_file,
    registered_file,
    graduated_file,
    supervisor_file,
    examiner_file,
])

if not all_files_uploaded:
    st.info("Please choose or drag and drop all five Excel files to continue.")
    st.stop()


data_raw = load_all_data(
    preprocess_file,
    registered_file,
    graduated_file,
    supervisor_file,
    examiner_file,
)

preprocess, registered, graduated, supervisor_students, examiner_detail = prepare_dataframes(data_raw)

st.success("All files uploaded successfully.")


# =========================================================
# SIMPLE OVERVIEW
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Pre-process", "Graduated", "Supervisors", "External Examiners"]
)

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-process", len(preprocess))
    c2.metric("Registered", len(registered))
    c3.metric("Graduated", len(graduated))

    if "Completion Year" in graduated.columns:
        grad_year = (
            graduated["Completion Year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .reset_index()
        )
        if not grad_year.empty:
            grad_year.columns = ["Completion Year", "Count"]
            fig = px.bar(grad_year, x="Completion Year", y="Count", title="Graduations by Year")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(preprocess, use_container_width=True)

with tab3:
    st.dataframe(graduated, use_container_width=True)

with tab4:
    st.dataframe(supervisor_students, use_container_width=True)

with tab5:
    st.dataframe(examiner_detail, use_container_width=True)
