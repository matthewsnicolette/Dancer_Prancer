import textwrap
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


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


def find_matching_column(df: pd.DataFrame, possibilities):
    lower_map = {c.lower(): c for c in df.columns}
    for item in possibilities:
        if item.lower() in lower_map:
            return lower_map[item.lower()]
    return None


def safe_columns(df: pd.DataFrame, columns):
    return [col for col in columns if col in df.columns]


def add_status_group_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Student Status Group" not in df.columns:
        if "Student Stage" in df.columns:
            stage = df["Student Stage"].astype(str).str.strip().str.lower()
            df["Student Status Group"] = np.where(
                stage == "graduated",
                "Graduated",
                "Currently in system",
            )
        else:
            df["Student Status Group"] = "Unknown"

    if "Status Detail" not in df.columns:
        if "Programme" in df.columns and "Completion Year" in df.columns:
            df["Status Detail"] = df.apply(
                lambda row: (
                    f"Graduated {row.get('Programme', '')} {row.get('Completion Year', '')}"
                    if row.get("Student Status Group") == "Graduated"
                    else f"Currently in system - {row.get('Programme', '')}"
                ),
                axis=1,
            )
        else:
            df["Status Detail"] = df["Student Status Group"]

    return df


def prepare_dataframes(data_dict):
    preprocess = clean_text_columns(standardize_columns(data_dict["preprocess"]))
    registered = clean_text_columns(standardize_columns(data_dict["registered"]))
    graduated = clean_text_columns(standardize_columns(data_dict["graduated"]))
    supervisor_students = clean_text_columns(standardize_columns(data_dict["supervisor_students"]))
    examiner_detail = clean_text_columns(standardize_columns(data_dict["examiner_detail"]))

    for col in ["Days in System", "Months in System", "Years in System"]:
        if col in registered.columns:
            registered[col] = pd.to_numeric(registered[col], errors="coerce")

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

    supervisor_students = add_status_group_if_missing(supervisor_students)
    examiner_detail = add_status_group_if_missing(examiner_detail)

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


def figure_to_rl_image(fig, width=9.8 * inch, height=5.3 * inch):
    try:
        img_bytes = fig.to_image(format="png", scale=2)
        img_buffer = BytesIO(img_bytes)
        pil_img = Image.open(img_buffer)
        img_width, img_height = pil_img.size
        aspect = img_height / img_width
        final_height = width * aspect

        if final_height > height:
            final_height = height
            width = final_height / aspect

        img_buffer.seek(0)
        return RLImage(img_buffer, width=width, height=final_height)

    except Exception:
        return None


def dataframe_preview_table(df, max_rows=15, max_cols=8, max_cell_len=28):
    if df is None or df.empty:
        return [["No data available"]]

    preview = df.head(max_rows).copy()

    if preview.shape[1] > max_cols:
        preview = preview.iloc[:, :max_cols].copy()

    preview = preview.fillna("")
    preview.columns = [
        textwrap.shorten(str(c), width=max_cell_len, placeholder="...")
        for c in preview.columns
    ]

    for col in preview.columns:
        preview[col] = preview[col].astype(str).apply(
            lambda x: textwrap.shorten(x, width=max_cell_len, placeholder="...")
        )

    return [preview.columns.tolist()] + preview.values.tolist()


def styled_report_table(df, col_widths=None):
    table = Table(df, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
            ]
        )
    )
    return table


def add_chart(story, title, fig, styles):
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Spacer(1, 0.12 * inch))

    chart_img = figure_to_rl_image(fig)

    if chart_img is not None:
        story.append(chart_img)
    else:
        story.append(
            Paragraph(
                "Chart image could not be embedded in this PDF in the current deployment environment. "
                "The interactive chart is still available in the Streamlit app.",
                styles["SmallNote"],
            )
        )

    story.append(Spacer(1, 0.25 * inch))


def build_pdf_report(
    preprocess_f,
    registered_f,
    graduated_f,
    supervisor_students_f,
    examiner_detail_f,
    workflow_col,
    assigned_supervisor_col,
    examiner_name_col,
    programme_color_map,
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#444444"),
        )
    )

    story = []

    story.append(Paragraph("Postgraduate Student Dashboard Report", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "This report was generated from uploaded Excel exports in the Streamlit dashboard.",
            styles["SmallNote"],
        )
    )
    story.append(Spacer(1, 0.25 * inch))

    summary_data = [
        ["Metric", "Count"],
        ["Pre-process Students", f"{len(preprocess_f):,}"],
        ["Registered Students", f"{len(registered_f):,}"],
        ["Graduated Students", f"{len(graduated_f):,}"],
    ]

    story.append(Paragraph("Overview Metrics", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(styled_report_table(summary_data, col_widths=[3.2 * inch, 1.2 * inch]))
    story.append(Spacer(1, 0.25 * inch))

    charts_made = False

    if "Completion Year" in graduated_f.columns:
        overview_grad = graduated_f.dropna(subset=["Completion Year"]).copy()
        if not overview_grad.empty:
            charts_made = True
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
                title="Number of Students Graduated by Year",
            )
            fig.update_xaxes(type="category")
            add_chart(story, "Overview: Number of Students Graduated by Year", fig, styles)

    if charts_made:
        story.append(PageBreak())

    story.append(Paragraph("Data Previews", styles["Heading1"]))
    story.append(Spacer(1, 0.12 * inch))

    preview_sections = [
        ("Pre-process Students", preprocess_f),
        ("Registered Students", registered_f),
        ("Graduated Students", graduated_f),
        ("Supervisor Detail", supervisor_students_f),
        ("External Examiner Detail", examiner_detail_f),
    ]

    for title, df in preview_sections:
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Spacer(1, 0.08 * inch))
        preview = dataframe_preview_table(df, max_rows=12)
        num_cols = len(preview[0])
        usable_width = 10.2 * inch
        col_width = usable_width / max(num_cols, 1)
        col_widths = [col_width] * num_cols
        story.append(styled_report_table(preview, col_widths=col_widths))
        story.append(Spacer(1, 0.22 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer


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
    preprocess_file = st.file_uploader("Pre-process students", type="xlsx", key="preprocess_file")
    registered_file = st.file_uploader("Registered students", type="xlsx", key="registered_file")
    graduated_file = st.file_uploader("Graduated students", type="xlsx", key="graduated_file")

with col2:
    supervisor_file = st.file_uploader("Supervisor student report", type="xlsx", key="supervisor_file")
    examiner_file = st.file_uploader("External examiner report", type="xlsx", key="examiner_file")

all_files_uploaded = all(
    [
        preprocess_file,
        registered_file,
        graduated_file,
        supervisor_file,
        examiner_file,
    ]
)

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
# GLOBAL FILTERS
# =========================================================
st.sidebar.subheader("Filters")

programme_options = sorted(
    set(preprocess.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(registered.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(graduated.get("Programme", pd.Series(dtype=object)).dropna().tolist())
)

selected_programmes = st.sidebar.multiselect("Programme", programme_options, default=[])


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if selected_programmes and "Programme" in out.columns:
        out = out[out["Programme"].isin(selected_programmes)]
    return out


preprocess_f = apply_filters(preprocess)
registered_f = apply_filters(registered)
graduated_f = apply_filters(graduated)
supervisor_students_f = apply_filters(supervisor_students)
examiner_detail_f = apply_filters(examiner_detail)


# =========================================================
# PROGRAMME COLOURS
# =========================================================
all_programmes = sorted(
    set(preprocess_f.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(registered_f.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(graduated_f.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(supervisor_students_f.get("Programme", pd.Series(dtype=object)).dropna().tolist())
    | set(examiner_detail_f.get("Programme", pd.Series(dtype=object)).dropna().tolist())
)

palette = px.colors.qualitative.Set3 + px.colors.qualitative.Bold + px.colors.qualitative.Safe
PROGRAMME_COLOR_MAP = {
    programme: palette[i % len(palette)]
    for i, programme in enumerate(all_programmes)
}


# =========================================================
# FIND IMPORTANT COLUMN NAMES
# =========================================================
workflow_col = find_matching_column(
    preprocess_f,
    ["Workflow Decision Status", "Workflow Status", "Status", "Decision Status"],
)

assigned_supervisor_col = find_matching_column(
    preprocess_f,
    ["Assigned Supervisor", "Supervisor", "Allocated Supervisor"],
)

examiner_name_col = find_matching_column(
    examiner_detail_f,
    ["External Examiner", "Examiner", "External Examiner Name"],
)


# =========================================================
# PDF DOWNLOAD
# =========================================================
try:
    pdf_bytes = build_pdf_report(
        preprocess_f=preprocess_f,
        registered_f=registered_f,
        graduated_f=graduated_f,
        supervisor_students_f=supervisor_students_f,
        examiner_detail_f=examiner_detail_f,
        workflow_col=workflow_col,
        assigned_supervisor_col=assigned_supervisor_col,
        examiner_name_col=examiner_name_col,
        programme_color_map=PROGRAMME_COLOR_MAP,
    )

    st.download_button(
        label="Download PDF report",
        data=pdf_bytes,
        file_name="student_dashboard_report.pdf",
        mime="application/pdf",
    )

except Exception:
    st.warning(
        "The PDF report could not include chart images in this deployment environment. "
        "The interactive charts in the app still work correctly."
    )


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Pre-process", "Registered", "Graduated", "Supervisors", "External Examiners"]
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
                title="Number of Students Graduated by Year",
            )
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PRE-PROCESS
# =========================================================
with tab2:
    st.subheader("Pre-process")

    if workflow_col and "Programme" in preprocess_f.columns:
        pp_programme_workflow = (
            preprocess_f.groupby(["Programme", workflow_col])
            .size()
            .reset_index(name="Students")
        )
        fig = px.bar(
            pp_programme_workflow,
            x="Programme",
            y="Students",
            color=workflow_col,
            barmode="group",
            title="Pre-process Students by Programme and Workflow Status",
        )
        st.plotly_chart(fig, use_container_width=True)

    if workflow_col and assigned_supervisor_col and "Programme" in preprocess_f.columns:
        pp_supervisor = preprocess_f.dropna(subset=[workflow_col]).copy()
        if not pp_supervisor.empty:
            pp_supervisor[assigned_supervisor_col] = pp_supervisor[assigned_supervisor_col].fillna("Unassigned")
            pp_sup_programme = (
                pp_supervisor.groupby([assigned_supervisor_col, "Programme"])
                .size()
                .reset_index(name="Students")
            )
            fig = px.bar(
                pp_sup_programme,
                x=assigned_supervisor_col,
                y="Students",
                color="Programme",
                color_discrete_map=PROGRAMME_COLOR_MAP,
                barmode="stack",
                title="Allocated Supervisors by Programme",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Pre-process Student Table")
    st.dataframe(preprocess_f, use_container_width=True)


# =========================================================
# REGISTERED
# =========================================================
with tab3:
    st.subheader("Registered")

    if "Programme" in registered_f.columns:
        reg_programme = (
            registered_f.groupby("Programme")
            .size()
            .reset_index(name="Students")
            .sort_values("Students", ascending=False)
        )

        fig = px.bar(
            reg_programme,
            x="Programme",
            y="Students",
            color="Programme",
            color_discrete_map=PROGRAMME_COLOR_MAP,
            title="Registered Students by Programme",
        )
        st.plotly_chart(fig, use_container_width=True)

    workflow_registered_col = find_matching_column(
        registered_f,
        ["Workflow Decision Status", "Workflow Status", "Workflow State", "Status", "Decision Status"],
    )

    if workflow_registered_col and "Programme" in registered_f.columns:
        reg_workflow = (
            registered_f.groupby(["Programme", workflow_registered_col])
            .size()
            .reset_index(name="Students")
        )

        fig = px.bar(
            reg_workflow,
            x="Programme",
            y="Students",
            color=workflow_registered_col,
            barmode="group",
            title="Registered Students by Programme and Workflow Status",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Registered Student Table")
    st.dataframe(registered_f, use_container_width=True)


# =========================================================
# GRADUATED
# =========================================================
with tab4:
    st.subheader("Graduated")

    if "Completion Year" in graduated_f.columns and "Programme" in graduated_f.columns:
        grad_chart = graduated_f.dropna(subset=["Completion Year"]).copy()
        if not grad_chart.empty:
            grad_chart["Completion Year"] = grad_chart["Completion Year"].astype(int)
            grad_by_year_programme = (
                grad_chart.groupby(["Completion Year", "Programme"])
                .size()
                .reset_index(name="Students")
                .sort_values("Completion Year")
            )
            fig = px.bar(
                grad_by_year_programme,
                x="Completion Year",
                y="Students",
                color="Programme",
                color_discrete_map=PROGRAMME_COLOR_MAP,
                barmode="stack",
                title="Graduated Students by Year and Programme",
            )
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Graduated Student Table")
    st.dataframe(graduated_f, use_container_width=True)


# =========================================================
# SUPERVISORS
# =========================================================
with tab5:
    st.subheader("Supervisors")

    if "Supervisor" in supervisor_students_f.columns and "Programme" in supervisor_students_f.columns:
        total_chart = (
            supervisor_students_f.groupby(["Supervisor", "Programme"])
            .size()
            .reset_index(name="Students")
        )
        fig_total = px.bar(
            total_chart,
            x="Supervisor",
            y="Students",
            color="Programme",
            color_discrete_map=PROGRAMME_COLOR_MAP,
            barmode="stack",
            title="Total Students per Supervisor by Programme",
        )
        st.plotly_chart(fig_total, use_container_width=True)

        st.markdown("### Individual Supervisor Profile")

        supervisor_options = sorted(supervisor_students_f["Supervisor"].dropna().unique())

        if supervisor_options:
            selected_supervisor = st.selectbox(
                "Select supervisor",
                supervisor_options,
                key="selected_supervisor_profile",
            )

            supervisor_profile = supervisor_students_f[
                supervisor_students_f["Supervisor"] == selected_supervisor
            ].copy()

            st.subheader(selected_supervisor)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Currently in system",
                int((supervisor_profile["Student Status Group"] == "Currently in system").sum())
                if "Student Status Group" in supervisor_profile.columns else 0,
            )
            c2.metric(
                "Graduated students",
                int((supervisor_profile["Student Status Group"] == "Graduated").sum())
                if "Student Status Group" in supervisor_profile.columns else 0,
            )
            c3.metric(
                "Primary students",
                int((supervisor_profile["Normalized Role"] == "Primary").sum())
                if "Normalized Role" in supervisor_profile.columns else 0,
            )
            c4.metric("Total students", len(supervisor_profile))

            if "Student Status Group" in supervisor_profile.columns:
                current_students = supervisor_profile[
                    supervisor_profile["Student Status Group"] == "Currently in system"
                ]
                graduated_students = supervisor_profile[
                    supervisor_profile["Student Status Group"] == "Graduated"
                ]

                st.markdown("#### Current students")
                st.dataframe(
                    current_students[
                        safe_columns(
                            current_students,
                            [
                                "Student Number",
                                "Student Name",
                                "Programme",
                                "Research Title",
                                "Role",
                                "Status Detail",
                            ],
                        )
                    ],
                    use_container_width=True,
                )

                st.markdown("#### Graduated students supervised")
                st.dataframe(
                    graduated_students[
                        safe_columns(
                            graduated_students,
                            [
                                "Student Number",
                                "Student Name",
                                "Programme",
                                "Research Title",
                                "Completion Year",
                                "Role",
                                "Status Detail",
                            ],
                        )
                    ],
                    use_container_width=True,
                )

                if "Completion Year" in graduated_students.columns and not graduated_students.empty:
                    st.markdown("#### Graduated students by year and programme")
                    sup_grad_year = (
                        graduated_students
                        .dropna(subset=["Completion Year"])
                        .groupby(["Completion Year", "Programme"])
                        .size()
                        .reset_index(name="Students")
                        .sort_values("Completion Year")
                    )
                    st.dataframe(sup_grad_year, use_container_width=True)

    st.markdown("### Supervisor Detail")
    st.dataframe(supervisor_students_f, use_container_width=True)


# =========================================================
# EXTERNAL EXAMINERS
# =========================================================
with tab6:
    st.subheader("External Examiners")

    if examiner_name_col and "Programme" in examiner_detail_f.columns:
        if "Student Status Group" in examiner_detail_f.columns:
            ex_current = examiner_detail_f[
                examiner_detail_f["Student Status Group"] == "Currently in system"
            ].copy()
            ex_graduated = examiner_detail_f[
                examiner_detail_f["Student Status Group"] == "Graduated"
            ].copy()
        else:
            ex_stage = examiner_detail_f.copy()
            ex_stage["Student Stage"] = ex_stage["Student Stage"].astype(str).str.strip().str.lower()
            ex_current = ex_stage[ex_stage["Student Stage"] == "registered"].copy()
            ex_graduated = ex_stage[ex_stage["Student Stage"] == "graduated"].copy()

        if not ex_current.empty:
            ex_current_chart = (
                ex_current.groupby([examiner_name_col, "Programme"])
                .size()
                .reset_index(name="Students")
            )
            fig = px.bar(
                ex_current_chart,
                x=examiner_name_col,
                y="Students",
                color="Programme",
                color_discrete_map=PROGRAMME_COLOR_MAP,
                barmode="stack",
                title="Current Students per External Examiner by Programme",
            )
            st.plotly_chart(fig, use_container_width=True)

        if not ex_graduated.empty:
            ex_grad_chart = (
                ex_graduated.groupby([examiner_name_col, "Programme"])
                .size()
                .reset_index(name="Students")
            )
            fig = px.bar(
                ex_grad_chart,
                x=examiner_name_col,
                y="Students",
                color="Programme",
                color_discrete_map=PROGRAMME_COLOR_MAP,
                barmode="stack",
                title="Graduated Students per External Examiner by Programme",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Individual External Examiner Profile")

        examiner_options = sorted(examiner_detail_f[examiner_name_col].dropna().unique())

        if examiner_options:
            selected_examiner = st.selectbox(
                "Select external examiner",
                examiner_options,
                key="selected_examiner_profile",
            )

            examiner_profile = examiner_detail_f[
                examiner_detail_f[examiner_name_col] == selected_examiner
            ].copy()

            st.subheader(selected_examiner)

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Currently in system",
                int((examiner_profile["Student Status Group"] == "Currently in system").sum())
                if "Student Status Group" in examiner_profile.columns else 0,
            )
            c2.metric(
                "Graduated students",
                int((examiner_profile["Student Status Group"] == "Graduated").sum())
                if "Student Status Group" in examiner_profile.columns else 0,
            )
            c3.metric("Total students examined", len(examiner_profile))

            if "Student Status Group" in examiner_profile.columns:
                current_examiner_students = examiner_profile[
                    examiner_profile["Student Status Group"] == "Currently in system"
                ]
                graduated_examiner_students = examiner_profile[
                    examiner_profile["Student Status Group"] == "Graduated"
                ]

                st.markdown("#### Current students")
                st.dataframe(
                    current_examiner_students[
                        safe_columns(
                            current_examiner_students,
                            [
                                "Student Number",
                                "Student Name",
                                "Programme",
                                "Research Title",
                                "Examiner Role",
                                "Status Detail",
                            ],
                        )
                    ],
                    use_container_width=True,
                )

                st.markdown("#### Graduated students examined")
                st.dataframe(
                    graduated_examiner_students[
                        safe_columns(
                            graduated_examiner_students,
                            [
                                "Student Number",
                                "Student Name",
                                "Programme",
                                "Research Title",
                                "Completion Year",
                                "Examiner Role",
                                "Status Detail",
                            ],
                        )
                    ],
                    use_container_width=True,
                )

                if "Completion Year" in graduated_examiner_students.columns and not graduated_examiner_students.empty:
                    st.markdown("#### Graduated students by year and programme")
                    examiner_grad_year = (
                        graduated_examiner_students
                        .dropna(subset=["Completion Year"])
                        .groupby(["Completion Year", "Programme"])
                        .size()
                        .reset_index(name="Students")
                        .sort_values("Completion Year")
                    )
                    st.dataframe(examiner_grad_year, use_container_width=True)

    st.markdown("### External Examiner Detail")
    st.dataframe(examiner_detail_f, use_container_width=True)
