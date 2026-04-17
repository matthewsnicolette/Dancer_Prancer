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


def find_matching_column(df: pd.DataFrame, possibilities):
    lower_map = {c.lower(): c for c in df.columns}
    for item in possibilities:
        if item.lower() in lower_map:
            return lower_map[item.lower()]
    return None


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
# DEGREE COLOURS
# =========================================================
DEGREE_COLOR_MAP = {
    "PhD": "#1f77b4",
    "MSc": "#ff7f0e",
    "M Agric": "#2ca02c",
    "Master's": "#d62728",
    "Other": "#9467bd",
    "Unknown": "#7f7f7f",
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
# OPTIONAL GLOBAL FILTERS
# =========================================================
st.sidebar.subheader("Filters")

programme_options = sorted(
    set(preprocess.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(registered.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(graduated.get("Programme", pd.Series(dtype=object)).dropna().tolist())
)

degree_options = sorted(
    set(preprocess.get("Degree Level", pd.Series(dtype=object)).dropna().tolist())
    | set(registered.get("Degree Level", pd.Series(dtype=object)).dropna().tolist())
    | set(graduated.get("Degree Level", pd.Series(dtype=object)).dropna().tolist())
)

selected_programmes = st.sidebar.multiselect("Programme", programme_options, default=[])
selected_degrees = st.sidebar.multiselect("Degree Level", degree_options, default=[])


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if selected_programmes and "Programme" in out.columns:
        out = out[out["Programme"].isin(selected_programmes)]

    if selected_degrees and "Degree Level" in out.columns:
        out = out[out["Degree Level"].isin(selected_degrees)]

    return out


preprocess_f = apply_filters(preprocess)
registered_f = apply_filters(registered)
graduated_f = apply_filters(graduated)
supervisor_students_f = apply_filters(supervisor_students)
examiner_detail_f = apply_filters(examiner_detail)


# =========================================================
# FIND IMPORTANT COLUMN NAMES
# =========================================================
workflow_col = find_matching_column(
    preprocess_f,
    ["Workflow Decision Status", "Workflow Status", "Status", "Decision Status"]
)

assigned_supervisor_col = find_matching_column(
    preprocess_f,
    ["Assigned Supervisor", "Supervisor", "Allocated Supervisor"]
)

examiner_name_col = find_matching_column(
    examiner_detail_f,
    ["External Examiner", "Examiner", "External Examiner Name"]
)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Pre-process", "Graduated", "Supervisors", "External Examiners"]
)


# =========================================================
# OVERVIEW
# =========================================================
with tab1:
    st.subheader("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-process", len(preprocess_f))
    c2.metric("Registered", len(registered_f))
    c3.metric("Graduated", len(graduated_f))

    if "Completion Year" in graduated_f.columns:
        overview_grad = graduated_f.dropna(subset=["Completion Year"]).copy()
        if not overview_grad.empty:
            overview_grad["Completion Year"] = overview_grad["Completion Year"].astype(int)

            overview_year = (
                overview_grad.groupby("Completion Year")
                .size()
                .reset_index(name="Students")
                .sort_values("Completion Year")
            )

            fig = px.bar(
                overview_year,
                x="Completion Year",
                y="Students",
                title="Number of Students Graduated by Year"
            )
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PRE-PROCESS
# =========================================================
with tab2:
    st.subheader("Pre-process")

    left, right = st.columns(2)

    with left:
        if workflow_col and "Degree Level" in preprocess_f.columns:
            pp_degree_workflow = (
                preprocess_f.groupby(["Degree Level", workflow_col])
                .size()
                .reset_index(name="Students")
            )

            fig = px.bar(
                pp_degree_workflow,
                x="Degree Level",
                y="Students",
                color=workflow_col,
                barmode="group",
                title="Pre-process Students by Degree and Workflow Status"
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if workflow_col and assigned_supervisor_col and "Degree Level" in preprocess_f.columns:
            pp_supervisor = preprocess_f.dropna(subset=[workflow_col]).copy()

            if not pp_supervisor.empty:
                pp_supervisor[assigned_supervisor_col] = pp_supervisor[assigned_supervisor_col].fillna("Unassigned")

                pp_sup_degree = (
                    pp_supervisor.groupby([assigned_supervisor_col, "Degree Level"])
                    .size()
                    .reset_index(name="Students")
                )

                fig = px.bar(
                    pp_sup_degree,
                    x=assigned_supervisor_col,
                    y="Students",
                    color="Degree Level",
                    color_discrete_map=DEGREE_COLOR_MAP,
                    barmode="stack",
                    title="Allocated Supervisors by Degree"
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Pre-process Student Table")
    st.dataframe(preprocess_f, use_container_width=True)


# =========================================================
# GRADUATED
# =========================================================
with tab3:
    st.subheader("Graduated")

    if "Completion Year" in graduated_f.columns and "Degree Level" in graduated_f.columns:
        grad_chart = graduated_f.dropna(subset=["Completion Year"]).copy()

        if not grad_chart.empty:
            grad_chart["Completion Year"] = grad_chart["Completion Year"].astype(int)

            grad_by_year_degree = (
                grad_chart.groupby(["Completion Year", "Degree Level"])
                .size()
                .reset_index(name="Students")
                .sort_values("Completion Year")
            )

            fig = px.bar(
                grad_by_year_degree,
                x="Completion Year",
                y="Students",
                color="Degree Level",
                color_discrete_map=DEGREE_COLOR_MAP,
                barmode="stack",
                title="Graduated Students by Year and Degree"
            )
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Graduated Student Table")
    st.dataframe(graduated_f, use_container_width=True)


# =========================================================
# SUPERVISORS
# =========================================================
with tab4:
    st.subheader("Supervisors")

    if "Supervisor" in supervisor_students_f.columns and "Degree Level" in supervisor_students_f.columns:
        supervisor_chart = (
            supervisor_students_f.groupby(["Supervisor", "Degree Level"])
            .size()
            .reset_index(name="Students")
        )

        fig = px.bar(
            supervisor_chart,
            x="Supervisor",
            y="Students",
            color="Degree Level",
            color_discrete_map=DEGREE_COLOR_MAP,
            barmode="stack",
            title="Students per Supervisor by Degree"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Supervisor Detail")
    st.dataframe(supervisor_students_f, use_container_width=True)


# =========================================================
# EXTERNAL EXAMINERS
# =========================================================
with tab5:
    st.subheader("External Examiners")

    if examiner_name_col and "Student Stage" in examiner_detail_f.columns and "Degree Level" in examiner_detail_f.columns:
        ex_stage = examiner_detail_f.copy()
        ex_stage["Student Stage"] = ex_stage["Student Stage"].astype(str).str.strip().str.lower()

        ex_graduated = ex_stage[ex_stage["Student Stage"] == "graduated"].copy()
        ex_registered = ex_stage[ex_stage["Student Stage"] == "registered"].copy()

        col1, col2 = st.columns(2)

        with col1:
            if not ex_graduated.empty:
                ex_grad_chart = (
                    ex_graduated.groupby([examiner_name_col, "Degree Level"])
                    .size()
                    .reset_index(name="Students")
                )

                fig = px.bar(
                    ex_grad_chart,
                    x=examiner_name_col,
                    y="Students",
                    color="Degree Level",
                    color_discrete_map=DEGREE_COLOR_MAP,
                    barmode="stack",
                    title="Graduated Students per External Examiner by Degree"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if not ex_registered.empty:
                ex_reg_chart = (
                    ex_registered.groupby([examiner_name_col, "Degree Level"])
                    .size()
                    .reset_index(name="Students")
                )

                fig = px.bar(
                    ex_reg_chart,
                    x=examiner_name_col,
                    y="Students",
                    color="Degree Level",
                    color_discrete_map=DEGREE_COLOR_MAP,
                    barmode="stack",
                    title="Registered Students per External Examiner by Degree"
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### External Examiner Detail")
    st.dataframe(examiner_detail_f, use_container_width=True)
