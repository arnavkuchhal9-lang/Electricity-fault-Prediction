import streamlit as st
import pandas as pd
import joblib


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="GridGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================
# SAFE CUSTOM CSS
# =====================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(6, 182, 212, 0.15),
                transparent 35%
            ),
            radial-gradient(
                circle at top right,
                rgba(139, 92, 246, 0.14),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #050816,
                #090f20
            );
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu,
    header,
    footer {
        visibility: hidden;
    }

    h1 {
        background: linear-gradient(
            90deg,
            #ffffff,
            #67e8f9,
            #a78bfa
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 55px !important;
        font-weight: 900 !important;
    }

    h2, h3 {
        color: #f8fafc !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(34, 211, 238, 0.20);
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.72);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stMetric"] {
        padding: 18px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 15px;
        background: rgba(2, 6, 23, 0.45);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    div[data-baseweb="select"] > div {
        min-height: 54px;
        border-radius: 13px;
        border-color: rgba(34, 211, 238, 0.35);
        background: rgba(2, 6, 23, 0.70);
    }

    .stButton > button {
        min-height: 54px;
        margin-top: 27px;
        border: none;
        border-radius: 13px;
        color: #06111b;
        font-size: 16px;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #22d3ee,
            #60a5fa,
            #a78bfa
        );
    }

    .stButton > button:hover {
        color: #06111b;
        border: none;
        box-shadow: 0 12px 32px rgba(34, 211, 238, 0.25);
        transform: translateY(-1px);
    }

    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(
            90deg,
            #22d3ee,
            #818cf8
        );
    }

    .stAlert {
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================
# LOAD FILES
# =====================================

area_fault_model = joblib.load(
    "area_fault_model.pkl"
)

recovery_model = joblib.load(
    "recovery_model.pkl"
)

area_encoder = joblib.load(
    "area_encoder.pkl"
)

fault_mapping = joblib.load(
    "fault_mapping.pkl"
)


# =====================================
# HEADER
# =====================================

st.caption("⚡ GRIDGUARD AI • POWER FAULT INTELLIGENCE")

st.title("Power Fault Intelligence")

st.write(
    "Muzaffarnagar ke kisi area ko select karke probable "
    "electricity faults aur estimated restoration time dekho."
)

st.success("● Prediction system operational")


# =====================================
# INPUT
# =====================================

st.subheader("Analyse affected area")

with st.container(border=True):

    input_column, button_column = st.columns(
        [2.5, 1]
    )

    with input_column:

        selected_area = st.selectbox(
            "Select affected area",
            area_encoder.classes_
        )

    with button_column:

        predict_button = st.button(
            "⚡ Analyse Fault",
            use_container_width=True
        )


# =====================================
# PREDICTION
# =====================================

if predict_button:

    try:
        area_code = area_encoder.transform(
            [selected_area]
        )[0]

        area_input = pd.DataFrame(
            [[area_code]],
            columns=["Main_Area"]
        )

        probabilities = (
            area_fault_model.predict_proba(
                area_input
            )[0]
        )

        fault_classes = area_fault_model.classes_

        results = []

        for fault_name, probability in zip(
            fault_classes,
            probabilities
        ):
            if probability <= 0:
                continue

            mapping = fault_mapping.get(
                fault_name
            )

            if mapping is None:
                continue

            recovery_input = pd.DataFrame(
                [[
                    area_code,
                    mapping["Fault_Level"],
                    mapping["Most_Likely_Fault"]
                ]],
                columns=[
                    "Main_Area",
                    "Fault_Level",
                    "Most_Likely_Fault"
                ]
            )

            predicted_minutes = (
                recovery_model.predict(
                    recovery_input
                )[0]
            )

            predicted_minutes = max(
                0,
                round(predicted_minutes)
            )

            hours = predicted_minutes // 60
            minutes = predicted_minutes % 60

            results.append({
                "fault": str(fault_name),
                "probability": probability * 100,
                "minutes": predicted_minutes,
                "time": f"{hours} hr {minutes} min"
            })

        results.sort(
            key=lambda result: result["probability"],
            reverse=True
        )

        top_results = results[:3]


        # =====================================
        # OUTPUT
        # =====================================

        if not top_results:

            st.warning(
                "Is area ke liye prediction available nahi hai."
            )

        else:

            st.subheader(
                f"Analysis for {selected_area}"
            )

            highest = top_results[0]

            st.info(
                f"⚡ Most likely fault: **{highest['fault']}**"
            )

            with st.container(border=True):

                st.caption(
                    "#1 MOST LIKELY FAULT"
                )

                st.subheader(
                    highest["fault"]
                )

                probability_column, time_column = (
                    st.columns(2)
                )

                with probability_column:

                    st.metric(
                        "Fault probability",
                        f"{highest['probability']:.2f}%"
                    )

                with time_column:

                    st.metric(
                        "Estimated restoration",
                        highest["time"]
                    )

                st.progress(
                    min(
                        highest["probability"] / 100,
                        1.0
                    )
                )


            # Other two faults
            if len(top_results) > 1:

                st.subheader(
                    "Other possible faults"
                )

                other_columns = st.columns(
                    len(top_results) - 1
                )

                for position, (
                    column,
                    result
                ) in enumerate(
                    zip(
                        other_columns,
                        top_results[1:]
                    ),
                    start=2
                ):

                    with column:

                        with st.container(
                            border=True
                        ):

                            st.caption(
                                f"#{position} POSSIBLE FAULT"
                            )

                            st.subheader(
                                result["fault"]
                            )

                            st.metric(
                                "Probability",
                                f"{result['probability']:.2f}%"
                            )

                            st.metric(
                                "Recovery time",
                                result["time"]
                            )

                            st.progress(
                                min(
                                    result["probability"] / 100,
                                    1.0
                                )
                            )


            # Weighted recovery estimate
            total_probability = sum(
                result["probability"]
                for result in top_results
            )

            if total_probability > 0:

                expected_minutes = round(
                    sum(
                        result["minutes"]
                        * result["probability"]
                        for result in top_results
                    ) / total_probability
                )

                expected_hours = (
                    expected_minutes // 60
                )

                remaining_minutes = (
                    expected_minutes % 60
                )

                st.success(
                    f"⏱️ Probability-weighted recovery estimate: "
                    f"**{expected_hours} hr "
                    f"{remaining_minutes} min**"
                )

    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "GridGuard AI • Predictions use simulated historical "
    "fault patterns for educational demonstration."
)