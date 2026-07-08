"""
app.py
------
Main Streamlit entry point for the AI-Powered Canteen Food Waste
Quantification System.

Run with:
    streamlit run app.py
"""

import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import (
    ensure_directories,
    PREDICTIONS_CSV,
    PROCESSED_IMAGES_DIR,
    CSS_PATH,
    CSV_COLUMNS,
    DEMO_CREDENTIALS,
    PROJECT_TITLE,
    PROJECT_SUBTITLE,
    INSTITUTION_NAME,
    CATEGORY_COLORS,
    new_record_id,
    now_timestamp,
    now_filename_safe,
    Timer,
)
from utils.preprocessing import run_preprocessing_pipeline
from utils.segmentation import segment_plate_and_food, detect_contours
from utils.feature_extraction import extract_features
from utils.predictor import predict_waste_category, compute_sustainability_score, load_model
from utils.recommendation import generate_recommendations
from utils.charts import (
    gauge_chart,
    category_pie_chart,
    category_bar_chart,
    trend_line_chart,
    daily_stats_bar,
    monthly_stats_bar,
    confidence_bar_chart,
)
from utils.pdf_export import generate_pdf_report

# --------------------------------------------------------------------------
# Page config & global setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="FoodWasteAI | Canteen Waste Quantification",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()


def load_css():
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# --------------------------------------------------------------------------
# Session state initialisation
# --------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "authenticated": False,
        "username": None,
        "role": None,
        "display_name": None,
        "page": "Dashboard",
        "last_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --------------------------------------------------------------------------
# CSV / history helpers
# --------------------------------------------------------------------------
def load_history() -> pd.DataFrame:
    if os.path.exists(PREDICTIONS_CSV):
        try:
            df = pd.read_csv(PREDICTIONS_CSV)
            for col in CSV_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[CSV_COLUMNS]
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame(columns=CSV_COLUMNS)


def append_history(record: dict):
    df = load_history()
    new_row = pd.DataFrame([{col: record.get(col) for col in CSV_COLUMNS}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(PREDICTIONS_CSV, index=False)


def save_history(df: pd.DataFrame):
    df.to_csv(PREDICTIONS_CSV, index=False)


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------
def metric_card(label, value, sub=None, col=None):
    target = col if col is not None else st
    badge = f"<div class='fw-metric-sub'>{sub}</div>" if sub else ""
    target.markdown(
        f"""
        <div class="fw-metric">
            <div class="fw-metric-label">{label}</div>
            <div class="fw-metric-value">{value}</div>
            {badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def category_badge(category):
    css_class = {
        "Low Waste": "fw-badge-green",
        "Moderate Waste": "fw-badge-amber",
        "High Waste": "fw-badge-red",
    }.get(category, "fw-badge-green")
    return f"<span class='fw-badge {css_class}'>{category}</span>"


# --------------------------------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------------------------------
def render_login():
    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown(
            f"""
            <div class="fw-card" style="text-align:center; margin-top: 60px;">
                <h1 style="margin-bottom:0;">🥗 FoodWasteAI</h1>
                <p style="color:#6B6B6B;">{PROJECT_SUBTITLE}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            st.markdown("#### Sign in to continue")
            username = st.text_input("Username", placeholder="admin or student")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                user = DEMO_CREDENTIALS.get(username.strip().lower())
                if user and user["password"] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username.strip().lower()
                    st.session_state.role = user["role"]
                    st.session_state.display_name = user["name"]
                    st.session_state.page = "Dashboard"
                    st.success(f"Welcome, {user['name']}!")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")

        with st.expander("🔑 Demo credentials"):
            st.markdown(
                """
                **Admin** — username: `admin` &nbsp;|&nbsp; password: `admin123`

                **Student** — username: `student` &nbsp;|&nbsp; password: `student123`
                """
            )


# --------------------------------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------------------------------
def render_dashboard():
    st.markdown(
        f"""
        <div class="fw-banner">
            <h1>Welcome back, {st.session_state.display_name} 👋</h1>
            <p>{PROJECT_TITLE} — turning plate photos into actionable sustainability insights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_history()
    total = len(df)
    avg_waste = df["waste_percentage"].mean() if total else 0
    high_count = int((df["predicted_category"] == "High Waste").sum()) if total else 0
    avg_score = df["sustainability_score"].mean() if total else 0

    c1, c2, c3, c4 = st.columns(4)
    metric_card("Total Analyses", total, "All-time scans", c1)
    metric_card("Avg Waste %", f"{avg_waste:.1f}%" if total else "—", "Across all plates", c2)
    metric_card("High-Waste Plates", high_count, "Needs attention", c3)
    metric_card("Avg Sustainability", f"{avg_score:.0f}/100" if total else "—", "Higher is better", c4)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 🧠 AI Workflow")
        st.markdown(
            """
            1. **Upload** a plate image (JPG / PNG)
            2. **Preprocess** — resize, denoise, blur, HSV convert
            3. **Segment** plate vs. leftover food (color + morphology + contours)
            4. **Extract features** — area, coverage, contour count, color stats
            5. **Classify** waste level with a trained RandomForestClassifier
            6. **Recommend** actions and score sustainability
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚡ Quick Actions")
        if st.button("📤 Analyze New Plate", use_container_width=True):
            st.session_state.page = "Analyze"
            st.rerun()
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.page = "Analytics"
            st.rerun()
        if st.button("🗂️ Browse History", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if total:
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 📈 Recent Trend")
        st.plotly_chart(trend_line_chart(df.tail(20)), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# ANALYZE PAGE (Upload -> CV pipeline -> ML prediction)
# --------------------------------------------------------------------------
def render_analyze():
    st.markdown("## 📤 Analyze a Plate Image")
    st.caption("Upload a photo of a plate after a meal to estimate leftover food waste.")

    uploaded_file = st.file_uploader(
        "Drag & drop or browse a plate image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload a JPG or PNG plate image to begin the analysis.")
        return

    pil_image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(pil_image)
    bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
    st.markdown("#### 🖼️ Uploaded Image Preview")
    st.image(pil_image, caption=uploaded_file.name, width=360)
    st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("🚀 Run AI Analysis", type="primary")

    if run:
        with st.spinner("Running computer vision pipeline and ML inference..."):
            with Timer() as timer:
                # --- Stage 1: Preprocessing ---
                stages = run_preprocessing_pipeline(bgr_image)

                # --- Stage 2: Segmentation ---
                seg = segment_plate_and_food(stages["hsv"], stages["resized"])
                contours, contour_image = detect_contours(seg["food_mask"])

                # --- Stage 3: Feature extraction ---
                features = extract_features(
                    stages["resized"], stages["hsv"], seg["plate_mask"], seg["food_mask"], contours
                )

                # --- Stage 4: ML prediction ---
                prediction = predict_waste_category(features)
                sustainability_score = compute_sustainability_score(features["waste_percentage"])
                recommendations = generate_recommendations(
                    prediction["category"], features["waste_percentage"], features
                )
            processing_time_ms = timer()

        st.success("Analysis complete!")

        # Save processed/segmented image for report + history
        img_filename = f"processed_{now_filename_safe()}.png"
        img_path = os.path.join(PROCESSED_IMAGES_DIR, img_filename)
        cv2.imwrite(img_path, seg["segmented_display"])

        # ---- Visualisation of CV pipeline ----
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 🔬 Computer Vision Pipeline Stages")
        tabs = st.tabs(["Original", "Preprocessed", "Segmented", "Contours"])
        with tabs[0]:
            st.image(cv2.cvtColor(stages["resized"], cv2.COLOR_BGR2RGB), width=420)
        with tabs[1]:
            st.image(cv2.cvtColor(stages["blurred"], cv2.COLOR_BGR2RGB), width=420)
        with tabs[2]:
            st.image(cv2.cvtColor(seg["segmented_display"], cv2.COLOR_BGR2RGB), width=420)
            st.caption("🟩 Plate region &nbsp;&nbsp; 🟥 Detected leftover food region")
        with tabs[3]:
            st.image(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB), width=420)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Prediction results ----
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Prediction Result")
        r1, r2, r3, r4 = st.columns(4)
        metric_card("Waste %", f"{features['waste_percentage']:.1f}%", col=r1)
        metric_card("Category", "", col=r2)
        r2.markdown(category_badge(prediction["category"]), unsafe_allow_html=True)
        metric_card("Confidence", f"{prediction['confidence']:.1f}%", col=r3)
        metric_card("Processing Time", f"{processing_time_ms:.0f} ms", col=r4)

        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.plotly_chart(gauge_chart(sustainability_score), use_container_width=True)
        with cc2:
            st.plotly_chart(confidence_bar_chart(prediction["probabilities"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Recommendations ----
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 💡 AI-Generated Recommendations")
        for tip in recommendations:
            st.markdown(f"- {tip}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Save to history ----
        record = {
            "id": new_record_id(),
            "timestamp": now_timestamp(),
            "image_name": uploaded_file.name,
            **features,
            "predicted_category": prediction["category"],
            "confidence": prediction["confidence"],
            "sustainability_score": sustainability_score,
            "processing_time_ms": round(processing_time_ms, 2),
        }
        append_history(record)

        record_with_tips = {**record, "recommendations": recommendations}
        st.session_state.last_result = {"record": record_with_tips, "image_path": img_path}

        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 📄 Export")
        e1, e2 = st.columns(2)
        with e1:
            if st.button("📑 Generate PDF Report", use_container_width=True):
                pdf_path = generate_pdf_report(record_with_tips, img_path)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=f.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True,
                    )
        with e2:
            with open(img_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Processed Image",
                    data=f.read(),
                    file_name=img_filename,
                    mime="image/png",
                    use_container_width=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# ANALYTICS PAGE
# --------------------------------------------------------------------------
def render_analytics():
    st.markdown("## 📊 Analytics Dashboard")
    df = load_history()

    if df.empty:
        st.info("No predictions yet. Analyze a plate image first to populate analytics.")
        return

    total = len(df)
    avg_waste = df["waste_percentage"].mean()
    max_waste = df["waste_percentage"].max()
    min_waste = df["waste_percentage"].min()

    c1, c2, c3, c4 = st.columns(4)
    metric_card("Total Analyses", total, col=c1)
    metric_card("Average Waste", f"{avg_waste:.1f}%", col=c2)
    metric_card("Highest Waste", f"{max_waste:.1f}%", col=c3)
    metric_card("Lowest Waste", f"{min_waste:.1f}%", col=c4)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 🥧 Category Distribution")
        st.plotly_chart(category_pie_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 📶 Avg Waste % by Category")
        st.plotly_chart(category_bar_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
    st.markdown("#### 📈 Waste Trend Over Time")
    st.plotly_chart(trend_line_chart(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 📅 Daily Statistics")
        st.plotly_chart(daily_stats_bar(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
        st.markdown("#### 🗓️ Monthly Statistics")
        st.plotly_chart(monthly_stats_bar(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HISTORY PAGE
# --------------------------------------------------------------------------
def render_history():
    st.markdown("## 🗂️ Prediction History")
    df = load_history()

    if df.empty:
        st.info("No prediction history yet.")
        return

    with st.expander("🔍 Search, Filter & Sort", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            search = st.text_input("Search by image name")
        with col2:
            category_filter = st.multiselect(
                "Filter by category", options=sorted(df["predicted_category"].dropna().unique())
            )
        with col3:
            sort_by = st.selectbox(
                "Sort by", ["timestamp", "waste_percentage", "confidence", "sustainability_score"]
            )

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["image_name"].str.contains(search, case=False, na=False)]
    if category_filter:
        filtered = filtered[filtered["predicted_category"].isin(category_filter)]
    filtered = filtered.sort_values(sort_by, ascending=False)

    st.dataframe(filtered, use_container_width=True, height=380)

    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered CSV", data=csv_bytes, file_name="foodwaste_history.csv",
            mime="text/csv", use_container_width=True,
        )
    with c2:
        if st.button("🗑️ Delete Filtered Rows", use_container_width=True):
            remaining = df.drop(filtered.index)
            save_history(remaining)
            st.success(f"Deleted {len(filtered)} record(s).")
            st.rerun()


# --------------------------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------------------------
def render_about():
    st.markdown("## ℹ️ About This Project")

    st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
    st.markdown("#### Problem Statement")
    st.write(
        "Canteens and mess facilities generate significant food waste daily, but this waste is "
        "rarely measured objectively. Manual estimation is subjective, slow, and inconsistent, "
        "making it difficult to plan interventions or track improvement over time."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
    st.markdown("#### Objectives")
    st.markdown(
        """
        - Automatically estimate leftover food percentage from a plate photo
        - Classify waste into Low / Moderate / High categories using ML
        - Provide dynamic, actionable recommendations
        - Track trends over time through an analytics dashboard
        - Generate exportable PDF reports for record keeping
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
    st.markdown("#### Methodology & AI Workflow")
    st.markdown(
        """
        **Computer Vision (OpenCV):** resize → denoise → Gaussian blur → HSV conversion →
        color-based plate/food segmentation → morphological cleanup → contour detection →
        feature extraction (area, coverage ratio, contour count, color statistics).

        **Machine Learning (scikit-learn):** a `RandomForestClassifier` (n_estimators=100)
        is trained on engineered features to classify waste level. A synthetic, statistically
        realistic dataset is auto-generated when a real labelled dataset is unavailable,
        ensuring the app is always runnable end-to-end.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
    st.markdown("#### Technology Stack")
    st.markdown(
        """
        | Layer | Technology |
        |---|---|
        | Frontend | Streamlit |
        | Backend | Python |
        | Computer Vision | OpenCV |
        | Machine Learning | scikit-learn (RandomForestClassifier) |
        | Visualization | Plotly, Matplotlib |
        | Data handling | NumPy, Pandas |
        | Report Export | ReportLab (PDF) |
        | Storage | CSV |
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fw-card'>", unsafe_allow_html=True)
    st.markdown("#### Future Enhancements")
    st.markdown(
        """
        - Deep-learning based segmentation (U-Net / Mask R-CNN) for finer-grained food masks
        - Food-type classification alongside waste quantification
        - Multi-camera live canteen deployment with real-time dashboards
        - Integration with kitchen inventory systems for closed-loop waste reduction
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🥗 FoodWasteAI")
        st.caption(INSTITUTION_NAME)
        st.markdown("---")
        st.markdown(f"**{st.session_state.display_name}**  \n`{st.session_state.role}`")
        st.markdown("---")

        pages = ["Dashboard", "Analyze", "Analytics", "History", "About"]
        icons = {"Dashboard": "🏠", "Analyze": "📤", "Analytics": "📊", "History": "🗂️", "About": "ℹ️"}
        choice = st.radio(
            "Navigation",
            pages,
            index=pages.index(st.session_state.page) if st.session_state.page in pages else 0,
            format_func=lambda p: f"{icons[p]}  {p}",
            label_visibility="collapsed",
        )
        st.session_state.page = choice

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["authenticated", "username", "role", "display_name"]:
                st.session_state[key] = None
            st.session_state.authenticated = False
            st.rerun()


# --------------------------------------------------------------------------
# MAIN ROUTER
# --------------------------------------------------------------------------
def main():
    if not st.session_state.authenticated:
        render_login()
        return

    render_sidebar()

    page = st.session_state.page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Analyze":
        render_analyze()
    elif page == "Analytics":
        render_analytics()
    elif page == "History":
        render_history()
    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()
