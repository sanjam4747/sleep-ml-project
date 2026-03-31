import streamlit as st
import pickle
import pandas as pd

# Load artifacts produced by training pipeline.
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURE_COLUMNS = [
    "WorkoutTime",
    "ReadingTime",
    "PhoneTime",
    "WorkHours",
    "CaffeineIntake",
    "RelaxationTime",
]


def generate_recommendations(workout, reading, phone, work, caffeine, relax, predicted_sleep):
    recommendations = []
    positives = []

    if phone > 6:
        target = 4
        reduction = phone - target
        recommendations.append(
            f"PhoneTime is {phone:.1f} h/day. Reduce by about {reduction:.1f} h (target: <= {target} h), "
            "especially in the last 1-2 hours before bed."
        )
    elif phone <= 4:
        positives.append(f"PhoneTime is well controlled at {phone:.1f} h/day.")

    if caffeine > 300:
        target = 200
        reduction = caffeine - target
        recommendations.append(
            f"CaffeineIntake is {caffeine:.0f} mg/day. Cut around {reduction:.0f} mg (target: <= {target} mg/day) "
            "and avoid caffeine after 2 PM."
        )
    elif caffeine <= 200:
        positives.append(f"CaffeineIntake is moderate at {caffeine:.0f} mg/day.")

    if work > 9:
        target = 8
        reduction = work - target
        recommendations.append(
            f"WorkHours is {work:.1f} h/day. Try to reduce by {reduction:.1f} h where possible (target: <= {target} h) "
            "or add short recovery breaks every 90-120 minutes."
        )

    if relax < 1.0:
        target = 1.5
        increase = target - relax
        recommendations.append(
            f"RelaxationTime is {relax:.1f} h/day. Add at least {increase:.1f} h of wind-down activities "
            "(reading, stretching, breathing) before sleep."
        )
    elif relax >= 1.5:
        positives.append(f"RelaxationTime is good at {relax:.1f} h/day.")

    if workout < 0.5:
        target = 1.0
        increase = target - workout
        recommendations.append(
            f"WorkoutTime is {workout:.1f} h/day. Add about {increase:.1f} h of light to moderate activity "
            f"(target: >= {target} h/day)."
        )
    elif workout >= 0.75:
        positives.append(f"WorkoutTime is supportive at {workout:.1f} h/day.")

    if reading < 0.3:
        target = 0.5
        increase = target - reading
        recommendations.append(
            f"ReadingTime is {reading:.1f} h/day. Add around {increase:.1f} h of low-stimulation reading "
            "in the evening to improve sleep onset."
        )

    if predicted_sleep < 6 and not recommendations:
        recommendations.append(
            "Predicted sleep is below 6 hours. Keep wake-up and bedtime fixed daily and increase total evening wind-down time."
        )

    return recommendations, positives


def build_analysis_summary(workout, reading, phone, work, caffeine, relax, predicted_sleep):
    stress_load = "high" if (work > 9 or phone > 6 or caffeine > 300) else "moderate"
    recovery_level = "low" if (relax < 1.0 or workout < 0.5) else "good"
    sleep_band = "low" if predicted_sleep < 6 else "healthy" if predicted_sleep <= 9 else "high"

    return (
        f"Predicted sleep is {predicted_sleep:.2f} h ({sleep_band}). "
        f"Current routine shows {stress_load} daily load and {recovery_level} recovery balance."
    )

st.title("💤 AI Sleep Recommendation System")

st.write("Enter your daily lifestyle details:")

# inputs
workout = st.number_input("Workout Time (hours/day)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
reading = st.number_input("Reading Time (hours/day)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
phone = st.number_input("Phone Time (hours/day)", min_value=0.0, max_value=16.0, value=4.0, step=0.1)
work = st.number_input("Work Hours (hours/day)", min_value=0.0, max_value=16.0, value=8.0, step=0.1)
caffeine = st.number_input("Caffeine Intake (mg/day)", min_value=0.0, max_value=1000.0, value=100.0, step=10.0)
relax = st.number_input("Relaxation Time (hours/day)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

total_time = workout + reading + phone + work + relax
st.caption(f"Total tracked time: {total_time:.1f} / 24.0 hours")

is_time_valid = total_time <= 24
if total_time > 24:
    st.error(
        f"Total tracked time is {total_time:.1f} hours, which exceeds 24 hours/day. "
        "Please adjust Workout, Reading, Phone, Work, or Relaxation time."
    )
elif total_time >= 22:
    st.warning(
        f"Your tracked time is {total_time:.1f} hours, which is close to 24 hours/day. "
        "Double-check entries for realistic planning."
    )

if st.button("Predict Sleep"):
    if not is_time_valid:
        st.error("Prediction skipped because total tracked time must be 24 hours/day or less.")
    else:
        input_df = pd.DataFrame(
            [[workout, reading, phone, work, caffeine, relax]],
            columns=FEATURE_COLUMNS,
        )
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # fix unrealistic values
        prediction = max(0, min(prediction, 12))

        st.subheader(f"🛌 Predicted Sleep: {prediction:.2f} hours")

        summary = build_analysis_summary(workout, reading, phone, work, caffeine, relax, prediction)
        st.markdown("### Lifestyle Analysis")
        st.info(summary)

        recommendations, positives = generate_recommendations(
            workout, reading, phone, work, caffeine, relax, prediction
        )

        st.markdown("### Personalized Recommendations")
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.success(
                "Your inputs look balanced across activity, screen time, caffeine, and recovery. "
                "Keep this routine consistent to maintain healthy sleep."
            )

        if positives:
            st.markdown("### What You Are Doing Well")
            for item in positives:
                st.markdown(f"- {item}")