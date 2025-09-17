import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import base64
import time
from PIL import Image
import io
import requests
import gc

# Set page config
st.set_page_config(
    page_title="BCI- EEG & Emotion Detection",
    page_icon="🧠",
    layout="wide"
)

# Label mappings
label_to_command = {0: "🔴 Backward", 1: "⬅️ Left", 2: "➡️ Right", 3: "🟢 Forward"}
emotion_labels = {0: "😠 NEGATIVE", 1: "😐 NEUTRAL", 2: "😊 POSITIVE"}
emotion_colors = {0: "#ff4b4b", 1: "#a3a3a3", 2: "#4bb543"}

# Load the models and scalers
@st.cache_resource
def load_models():
    try:
        # Load EEG model
        with open('best_eeg_model.joblib', 'rb') as f:
            eeg_model = joblib.load(f)
        
        with open('eeg_scaler.joblib', 'rb') as f:
            eeg_scaler = joblib.load(f)
        
        # Load emotion model and preprocessing tools
        with open('best_emotion_model.joblib', 'rb') as f:
            emotion_model = joblib.load(f)
            
        with open('emotion_scaler.joblib', 'rb') as f:
            emotion_scaler = joblib.load(f)
        
        # Load feature selector for emotion detection
        with open('emotion_feature_selector.joblib', 'rb') as f:
            feature_selector = joblib.load(f)
        
        # Force garbage collection
        gc.collect()
        
        return eeg_model, eeg_scaler, emotion_model, emotion_scaler, feature_selector
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Function to load EEG sensor positions image
@st.cache_resource
def load_eeg_sensor_image():
    try:
        response = requests.get("https://hebbkx1anhila5yf.public.blob.vercel-storage.com/EEG_sensor_positions-4JF0gqWXRy6wxFkLC1FqD3SouEGsjw.png")
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error loading EEG sensor image: {str(e)}")
        return None

# Function to preprocess data for EEG
def preprocess_eeg_data(df):
    columns_to_drop = ['Start Timestamp', 'End Timestamp', 'Label']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns=columns_to_drop)
    return X

# Function to preprocess data for emotion detection with memory optimization
def preprocess_emotion_data(df):
    try:
        # Normalize the features
        X = df.copy()
        if 'Label' in X.columns:
            X = X.drop('Label', axis=1)
        
        # Convert to float32 to reduce memory usage
        X = X.astype(np.float32)
        return X
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

# Function to make EEG predictions
def make_eeg_predictions(model, scaler, X):
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions

# Function to make emotion predictions with memory optimization
def make_emotion_predictions(model, scaler, feature_selector, X):
    try:
        # Process in smaller batches to manage memory
        batch_size = 16  # Reduced batch size
        predictions = []
        
        # Get the number of features expected by the model
        model_features = 489  # From the error message

        print("---predictions-----------------------------1111")
        print("---len(X)-----------------------------", len(X))
        
        # Store original column names for debugging
        original_columns = X.columns.tolist()
        
        # Check if we're dealing with numbered features (Feature_0, Feature_1, etc.)
        # instead of the expected feature names (mean_0_a, correlate_0_a, etc.)
        has_numbered_features = any(col.startswith('Feature_') for col in original_columns)
        
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i:i+batch_size]
            
            # Scale the data first - convert to numpy array to avoid feature name issues
            batch_array = batch.values
            batch_scaled = scaler.transform(batch_array)
            
            # If the feature selector doesn't work, we need to ensure we're passing
            # the right number of features to the model
            try:
                # Try using the feature selector first
                # Since we're using numpy arrays now, feature names don't matter
                batch_features = feature_selector.transform(batch_scaled)
            except Exception as selector_error:
                print(f"Feature selector error: {str(selector_error)}")
                # Since the model expects 489 features but we may have more,
                # we'll need to select only the features the model was trained on
                if batch_scaled.shape[1] > model_features:
                    # Select only the first model_features columns
                    batch_features = batch_scaled[:, :model_features]
                else:
                    # If we have fewer features than expected, can't proceed
                    st.error(f"Model expects {model_features} features but input has only {batch_scaled.shape[1]}")
                    return None
            
            # Make predictions on batch
            batch_pred = model.predict_proba(batch_features)
            
            # Get class with highest probability
            if batch_pred.shape[1] == 3:  # If we have 3 emotion classes
                batch_classes = np.argmax(batch_pred, axis=1)
            else:
                # Assuming binary classification between negative (0) and positive (2)
                # with neutral (1) not being predicted
                batch_classes = np.where(batch_pred[:, 1] > 0.5, 2, 0)
            
            predictions.extend(batch_classes)
            
            # Clear memory after each batch
            del batch
            gc.collect()
        
        print("---predictions-----------------------------1111", predictions)
        return np.array(predictions)
    except Exception as e:
        st.error(f"Error making emotion predictions: {str(e)}")
        print(f"Exception details: {type(e).__name__}: {str(e)}")
        return None

# Function to make predictions
def make_predictions(model, scaler, X):
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions

# Function to create download link
def get_table_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Function to create emotion gauge chart
def create_emotion_gauge(emotion_counts):
    total = sum(emotion_counts.values())
    fig = go.Figure()
    
    for emotion, count in emotion_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=percentage,
            title={'text': emotion_labels[emotion]},
            domain={'x': [0.1, 0.9], 'y': [0.1 + (emotion * 0.3), 0.3 + (emotion * 0.3)]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': emotion_colors[emotion]},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
            }
        ))
    
    fig.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=30, b=30),
        showlegend=False,
        paper_bgcolor="white",
        font={'size': 16}
    )
    
    return fig

# Main app
def main():
    st.title("🧠 BCI- EEG & Emotion Detection")
    st.write("This app classifies EEG signals and detects emotions using trained models.")

    # Load models
    eeg_model, eeg_scaler, emotion_model, emotion_scaler, feature_selector = load_models()
    
    if eeg_model is None or eeg_scaler is None or emotion_model is None:
        st.error("Failed to load one or more models. Please check the model files and try again.")
        return

    # Load EEG sensor positions image
    eeg_sensor_image = load_eeg_sensor_image()

    # Add custom CSS for button styles
    st.markdown("""
    <style>
        .stButton > button {
            width: 100%;
            height: 60px;
            font-size: 1.2em;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            background-color: #ffffff !important;
            border: 1px solid #ccc;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .emotion-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
            font-size: 1.2em;
        }
        .error-message {
            color: #ff4b4b;
            padding: 10px;
            border-radius: 5px;
            background-color: #ff4b4b20;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["EEG Batch Prediction", "Emotion Batch Prediction", "Single Signal Prediction", "Single Emotion Prediction"])

    with tab1:
        # EEG Batch Prediction content
        batch_cols = st.columns([7, 3])
        
        with batch_cols[0]:
            st.header("EEG Batch Prediction")
            st.write("Upload a CSV file containing EEG signals for batch prediction.")

            # Add sample data download
            st.markdown("### Sample Data")
            try:
                sample_data = pd.read_csv('sample_batch_data.csv')
                st.markdown(get_table_download_link(sample_data, 'sample_batch_data.csv'), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")

            uploaded_file = st.file_uploader("Choose a CSV file for EEG", type="csv", key="eeg_uploader")

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())

                    X = preprocess_eeg_data(df)
                    if st.button("Make EEG Predictions"):
                        with st.spinner("Processing EEG signals..."):
                            predictions = make_eeg_predictions(eeg_model, eeg_scaler, X)
                            
                            if predictions is not None:
                                results_df = df.copy()
                                results_df['Predicted_Label'] = predictions
                                results_df['Command'] = results_df['Predicted_Label'].map(label_to_command)

                                st.subheader("EEG Prediction Results")
                                st.dataframe(results_df)

                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download EEG Predictions",
                                    data=csv,
                                    file_name='eeg_predictions.csv',
                                    mime='text/csv'
                                )

                                st.subheader("Prediction Distribution")
                                fig = px.histogram(results_df, x='Command',
                                                 title='Distribution of Predicted Commands')
                                st.plotly_chart(fig)
                            else:
                                st.error("Failed to make predictions. Please try again.")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        # Right column for EEG Sensor Positions
        with batch_cols[1]:
            st.markdown("""
            <h2 style="font-size: 1.8rem; font-weight: 600; margin-bottom: 0.5rem;">EEG Sensor Positions</h2>
            """, unsafe_allow_html=True)

            with st.container():
                st.markdown("""
                <div class="sensor-container">
                    <div class="sensor-subtitle">International 10-20 System</div>
                </div>
                """, unsafe_allow_html=True)

                if eeg_sensor_image:
                    st.image(eeg_sensor_image, output_format="PNG", width=250)

    with tab2:
        # Emotion Batch Prediction content
        emotion_cols = st.columns([7, 3])
        
        with emotion_cols[0]:
            st.header("Emotion Batch Prediction")
            st.write("Upload a CSV file containing EEG signals for emotion prediction.")

            # Add sample data download
            st.markdown("### Sample Data")
            try:
                sample_emotion = pd.read_csv('sample_emotion.csv')
                st.markdown(get_table_download_link(sample_emotion, 'sample_emotion.csv'), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")

            uploaded_file = st.file_uploader("Choose a CSV file for Emotion Detection", type="csv", key="emotion_uploader")

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())

                    X = preprocess_emotion_data(df)
                    if st.button("Make Emotion Predictions"):
                        with st.spinner("Processing emotions..."):
                            predictions = make_emotion_predictions(emotion_model, emotion_scaler, feature_selector, X)
                            
                            if predictions is not None:
                                results_df = df.copy()
                                results_df['Predicted_Emotion'] = predictions
                                results_df['Emotion'] = results_df['Predicted_Emotion'].map(emotion_labels)

                                st.subheader("Emotion Prediction Results")
                                st.dataframe(results_df)

                                # Download predictions
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Emotion Predictions",
                                    data=csv,
                                    file_name='emotion_predictions.csv',
                                    mime='text/csv'
                                )

                                # Create emotion distribution visualization
                                emotion_counts = results_df['Predicted_Emotion'].value_counts().to_dict()
                                
                                # Ensure all emotions are represented
                                for emotion in emotion_labels.keys():
                                    if emotion not in emotion_counts:
                                        emotion_counts[emotion] = 0

                                st.subheader("Emotion Distribution")
                                gauge_fig = create_emotion_gauge(emotion_counts)
                                st.plotly_chart(gauge_fig, use_container_width=True)

                                # Show emotion timeline
                                st.subheader("Emotion Timeline")
                                timeline_fig = px.line(
                                    results_df.reset_index(),
                                    x='index',
                                    y='Predicted_Emotion',
                                    title='Emotion Changes Over Time',
                                    labels={'index': 'Sample Number', 'Predicted_Emotion': 'Emotion'},
                                    color_discrete_sequence=[emotion_colors[1]]
                                )
                                timeline_fig.update_layout(yaxis=dict(ticktext=list(emotion_labels.values()),
                                                                    tickvals=list(emotion_labels.keys())))
                                st.plotly_chart(timeline_fig)
                            else:
                                st.error("Failed to make predictions. Please try again.")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        # Right column for emotion information
        with emotion_cols[1]:
            st.markdown("""
            <h2 style="font-size: 1.8rem; font-weight: 600; margin-bottom: 0.5rem;">Emotion Categories</h2>
            """, unsafe_allow_html=True)

            for emotion_id, label in emotion_labels.items():
                st.markdown(f"""
                <div class="emotion-box" style="background-color: {emotion_colors[emotion_id]}20; border: 2px solid {emotion_colors[emotion_id]}">
                    {label}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            ### About Emotion Detection
            The emotion detection model analyzes EEG signals to classify emotional states into three categories:
            - 😊 **Positive**: Happy, excited, or content states
            - 😐 **Neutral**: Calm or balanced emotional states
            - 😠 **Negative**: Stressed, frustrated, or upset states
            
            The model uses various EEG features including:
            - Power band ratios (Alpha, Beta, Gamma)
            - Statistical measures
            - Frequency domain features
            """)

    with tab3:
        st.header("Single Signal Prediction")

        # Create a single column for input and prediction result
        input_col = st.container()

        with input_col:
            st.write("Click the below buttons to get predictions.")

            # Create buttons for each direction above the text area
            col1, col2, col3, col4 = st.columns(4)  # Create four columns for the buttons

            if 'pasted_data' not in st.session_state:
                st.session_state.pasted_data = ""

            with col1:
                if st.button("⬅️"):
                    st.session_state.pasted_data = "12.941,13.925,0.6384,10.47,5.69507750956912,5.730747106617077,-0.6799478080579413,-0.5096076385895788,23.78,161.44000000000003,283.2301102543645,535.8684057938913,122.84759251428356,757.493569136136,253.4486183671476,79.8,131.81576884959443,194.79125209214985,370.9546482454237,59.93189994946399,89.09799810426821,55.24933294773325,24.86027344873845,24.309113916943712,100.42302432769688,90.32092342764348,10.85806718874554,57.31898139333484,47.020354909146,67.64011704806995,18.10636717029908,78.98041804595647,30.3558967577572,59.4786643680694,10.922036293820709,44.5779904652658,29.437622539853944,36.05712785609055,14.347660669690088,5.792537390504529,6.797032729507975,11.944572227237176,2.103552362594324,4.660091628087092,4.62346371107315,5.70391822753298,3.925774688274664,3.700859356470557,4.056848371028937,4.741094330910339,4.563123522756883,4.119646171547212,3.814004389982326,3.890024903357447,3.481816369701896,3.9747004808916495,3.750845093219823,4.061976044838742,3.557365181930646,3.682522269370368,3.3782282674037063,3.768286880229158,3.4267676044531163,3.465072273168813,3.767530754360707,3.0512859705372515,3.563266785166551,3.5229448100731555,3.5044338404928923,3.299954363057375,3.1170446627813537,3.4812779750409564,3.2402856971806333,3.2599130450828198,3.581453526862449,3.6143937199739455,3.451437312299841,3.329495624304338,3.329495624304338,3.451437312299841,3.6143937199739455,3.581453526862449,3.2599130450828198,3.2402856971806333,3.4812779750409564,3.1170446627813537,3.299954363057375,3.5044338404928923,3.5229448100731555,3.563266785166551,3.0512859705372515,3.767530754360707,3.465072273168813,3.4267676044531163,3.768286880229158,3.3782282674037063,3.682522269370368,3.557365181930646,4.061976044838742,3.750845093219823,3.9747004808916495,3.481816369701896,3.890024903357447,3.814004389982326,4.119646171547212,4.563123522756883,4.741094330910339,4.056848371028937,3.700859356470557,3.925774688274664,5.70391822753298,4.62346371107315,4.660091628087092,2.103552362594324,11.944572227237176,6.797032729507975,5.792537390504529,14.347660669690088,36.05712785609055,29.437622539853944,44.5779904652658,10.922036293820709,59.4786643680694,30.3558967577572,78.98041804595647,18.10636717029908,67.64011704806995,47.020354909146,57.31898139333484,10.85806718874554,90.32092342764348,100.42302432769688,24.309113943712,24.86027344873845,55.24933294773325,89.09799810426821,59.93189994946399,370.9546482454237,194.79125209214985,131.81576884959443"

            with col2:
                if st.button("➡️"):
                    st.session_state.pasted_data = "17.395,18.388,0.42024,5.22,1.6131343224914656,1.6669745049040192,0.5025110070067309,0.239533240624632,9.1,92.08,165.4295971944718,309.57432302160333,58.24719610414238,84.92850314212005,172.0875267674643,52.53,4.349501719306558,4.28840771540847,39.862175538162425,36.4284181692426,30.46551890972791,25.01970495571268,46.3652031959139,33.808681536867226,38.296132245945,12.626765084859084,49.2025937833613,31.4954245434392,52.86057975123696,24.55065196899352,23.27063818626752,32.13744783420577,21.91764509090947,23.78136989731783,15.568329716878544,8.14584129140285,14.028810047107587,25.893231464828432,7.516908722403525,2.595273725776378,5.048236693896485,6.387403956953162,4.361018630229035,2.177201992767261,5.341293038851984,2.497016468137817,3.344512943382261,4.280914528765986,1.9053203907837477,2.2983513828834825,1.3831502596286656,2.00990638700909,1.5212893856644114,2.0075263733257755,2.1062263604341616,1.6701857137564855,1.4578214827166478,1.9360366404920484,1.5368561570693156,1.735988854665956,1.5867914113761434,1.584388961931582,1.7926898224178711,1.4874799944259034,2.1135601963672013,1.3390691850674177,1.361720566377774,1.2895449036330602,1.342126487978718,1.4518638252653944,1.6633728850614524,1.5582813919379872,1.4686284837006685,1.42874847968582,1.085152993841838,1.1811843939986342,1.3089963739920012,1.5124924183670656,1.5124924183670656,1.3089963739920012,1.1811843939986342,1.085152993841838,1.42874847968582,1.4686284837006685,1.5582813919379872,1.6633728850614524,1.4518638252653944,1.342126487978718,1.2895449036330602,1.361720566377774,1.3390691850674177,2.1135601963672013,1.4874799944259034,1.7926898224178711,1.584388961931582,1.5867914113761434,1.735988854665956,1.5368561570693156,1.9360366404920484,1.4578214827166478,1.6701857137564855,2.1062263604341616,2.0075263733257755,1.5212893856644114,2.00990638700909,1.3831502596286656,2.2983513828834825,1.9053203907837477,4.280914528765986,3.344512943382261,2.497016468137817,5.341293038851984,2.177201992767261,4.361018630229035,6.387403956953162,5.048236693896485,2.595273725776378,7.516908722403525,25.893231464828432,14.028810047107587,8.14584129140285,15.568329716878544,23.78136989731783,21.91764509090947,32.13744783420577,23.27063818626752,24.55065196899352,52.86057975123696,31.4954245434392,49.2025937833613,12.626765084859084,38.296132245945,33.808681536867226,46.3652031959139,25.01970495571268,30.46551890972791,36.4284181692426,39.862175538162425,4.28840771540847,4.349501719306558"

            with col3:
                if st.button("🟢"):
                    st.session_state.pasted_data = "13.925,14.924,1.87152,10.78,3.1565239250796115,3.669636276254092,0.1211691859223678,0.6603649862081264,14.81,132.56,198.80622450046425,407.0805420038367,41.30326758504787,377.607150917067,329.6929704450757,233.94,148.7013816289654,85.91753652415252,69.78750390743814,73.20072885651093,73.1322894014404,98.02996518368482,34.041009405378574,51.28897759806096,37.05852623857877,42.71157200488174,4.121505249611095,63.62564340933166,31.534858757085843,26.046172390334792,31.967841793106,35.26869307443615,16.832909786511177,14.58838382535911,72.71765423745356,39.38033547082446,13.050879655426137,10.622103786120103,16.61259155569531,7.327163401325991,2.697414899833233,9.808442155880304,4.763893431803291,1.933088662754356,5.737719153982554,2.5647525565726794,2.712132672587128,1.7239571826365292,1.6810039195561426,1.2736319462766283,2.1208764527542963,1.431964899448816,1.5971458512336798,1.2528988180493683,0.9752943870497016,1.3979536027060762,1.1679119156365092,1.1380472712930905,1.0470923307776576,0.8319348985150075,1.05327854458485,1.023053339527435,1.0215374156567398,1.2524345353520694,1.1490328654912667,1.001824764196653,1.068374193811373,1.0746584124407246,1.130687521124672,0.93008616766653,1.0557527635382242,0.6299639484264786,0.5573043439644885,0.7358360777936935,1.1573668875401568,1.3547238103124235,1.0282593944166858,1.1624938941100904,1.1624938941100904,1.0282593944166858,1.3547238103124235,1.1573668875401568,0.7358360777936935,0.5573043439644885,0.6299639484264786,1.0557527635382242,0.93008616766653,1.130687521124672,1.0746584124407246,1.068374193811373,1.001824764196653,1.1490328654912667,1.2524345353520694,1.0215374156567398,1.023053339527435,1.05327854458485,0.8319348985150075,1.0470923307776576,1.1380472712930905,1.1679119156365092,1.3979536027060762,0.9752943870497016,1.2528988180493683,1.5971458512336798,1.431964899448816,2.1208764527542963,1.2736319462766283,1.6810039195561426,1.7239571826365292,2.712132672587128,2.5647525565726794,5.737719153982554,1.933088662754356,4.763893431803291,9.808442155880304,2.697414899833233,7.327163401325991,16.61259155569531,10.622103786120103,13.050879655426137,39.38033547082446,72.71765423745356,14.58838382535911,16.832909786511177,35.26869307443615,31.967841793106,26.046172390334792,31.534858757085843,63.62564340933166,4.121505249611095,42.71157200488174,37.05852623857877,51.28897759806096,34.041009405378574,98.02996518368482,73.1322894014404,73.20072885651093,69.78750390743814,85.91753652415252,148.7013816289654"

            with col4:
                if st.button("🔴"):
                    st.session_state.pasted_data = "17.895,18.892,0.5593599999999999,4.19,1.3399031272446529,1.451972451529298,0.0320338884493001,0.2706543858059262,6.720000000000001,76.88,140.49172965028924,242.71788485255345,14.893473125791424,92.5938578955408,158.00432313596815,69.92,17.648413440914005,17.60878433820803,15.942900926788193,41.39375918963057,31.65929981399516,28.51668016333828,11.837738400891023,44.59684556811312,23.26107343091225,31.31308591692754,26.198424253335173,15.122300481001137,11.947044300968017,35.65477050898056,40.74540871709828,3.170204831248526,17.743787859483145,7.832579625232223,14.202732782976067,11.234295213345131,25.86605384164161,10.297599158678702,15.900881342063846,10.847474399616363,1.2189751711523162,6.250109488078362,7.004108841778587,2.070034778184759,3.987326881330624,1.6221966296951873,1.8513863201267733,1.0579632953423768,0.6048262688858255,1.1045783463260963,0.2327169453534671,0.1513056928571692,0.5569245010654496,0.3110401781217685,0.5506589456930167,0.4080679794684767,0.1097516947495553,0.1624563203280561,0.3824027368042154,0.2869173274904243,0.497512987331642,0.323380254486268,0.2630683124703888,0.4477154213611007,0.4652196010429258,0.5571351111841539,0.0248571056015484,0.2282924971433172,0.2121943384031659,0.39426722460893,0.1437967244183503,0.3457206082414431,0.0580132488613787,0.1442489900632044,0.3821711343209582,0.2631918582884829,0.4013443380081823,0.3481501876481251,0.3481501876481251,0.4013443380081823,0.2631918582884829,0.3821711343209582,0.1442489900632044,0.0580132488613787,0.3457206082414431,0.1437967244183503,0.39426722460893,0.2121943384031659,0.2282924971433172,0.0248571056015484,0.5571351111841539,0.4652196010429258,0.4477154213611007,0.2630683124703888,0.323380254486268,0.497512987331642,0.2869173274904243,0.3824027368042154,0.1624563203280561,0.1097516947495553,0.4080679794684767,0.5506589456930167,0.3110401781217685,0.5569245010654496,0.1513056928571692,0.2327169453534671,1.1045783463260963,0.6048262688858255,1.0579632953423768,1.8513863201267733,1.6221966296951873,3.987326881330624,2.070034778184759,7.004108841778587,6.250109488078362,1.2189751711523162,10.847474399616363,15.900881342063846,10.297599158678702,25.86605384164161,11.234295213345131,14.202732782976067,7.832579625232223,17.743787859483145,3.170204831248526,40.74540871709828,35.65477050898056,11.947044300968017,15.122300481001137,26.198424253335173,31.31308591692754,23.26107343091225,44.59684556811312,11.837738400891023,28.51668016333828,31.65929981399516,41.39375918963057,15.942900926788193,17.60878433820803,17.648413440914005"
            # Create the text area below the buttons
            pasted_data = st.text_area("Paste a row of data (comma-separated values) to get predictions.", value=st.session_state.pasted_data, height=100)

            # Add Predict button
            predict_button = st.button("Predict")

            # Create a placeholder for prediction results
            prediction_result_placeholder = st.empty()

        # Create a container for visualization that will appear below the prediction result
        visualization_container = st.container()

        if pasted_data and predict_button:
            try:
                # Split the pasted data and create a DataFrame
                values = [float(x.strip()) for x in pasted_data.split(',')]
                feature_names = ['Start Timestamp', 'End Timestamp', 'Mean', 'Max', 'Standard Deviation', 'RMS',
                            'Kurtosis', 'Skewness', 'Peak-to-Peak', 'Abs Diff Signal', 'Alpha Power',
                            'Beta Power', 'Gamma Power', 'Delta Power', 'Theta Power'] + \
                            [f'FFT_{i}' for i in range(125)]

                if len(values) == len(feature_names):
                    input_data = dict(zip(feature_names, values))
                    input_df = pd.DataFrame([input_data])

                    # Make prediction
                    prediction = make_predictions(eeg_model, eeg_scaler, preprocess_eeg_data(input_df))

                    # Display prediction result - stacked as in the screenshot
                    with prediction_result_placeholder.container():
                        st.subheader("Prediction Result")
                        predicted_label = prediction[0]
                        command = label_to_command.get(predicted_label, "Unknown Command")

                        # Display results stacked (one below the other)
                        st.write(f"Predicted Label: {predicted_label}")

                        # Make command text slightly bigger using HTML
                        command_html = f"""
                        <div style="font-size: 1.2em; margin-top: 5px;">
                            <strong>Command:</strong> {command}
                        </div>
                        """
                        st.markdown(command_html, unsafe_allow_html=True)

                        st.markdown("---")  # Add a separator

                    # Create side-by-side visualization below the prediction result
                    with visualization_container:
                        # Use columns with specific width ratio (3:7) - 30% for Movement and 70% for Signal
                        viz_cols = st.columns([3, 7])

                        # Left column for wheelchair visualization - fixed size as in original but aligned left
                        with viz_cols[0]:
                            st.subheader("Movement Visualization")

                            # Define CSS for the animated wheelchair - keeping original size but aligned left
                            css = """
                            <style>
                            @keyframes moveLeft {
                                from { transform: translate(-50%, -50%); }
                                to { transform: translate(-150%, -50%); }
                            }
                            @keyframes moveRight {
                                from { transform: translate(-50%, -50%); }
                                to { transform: translate(50%, -50%); }
                            }
                            @keyframes moveForward {
                                from { transform: translate(-50%, -50%); }
                                to { transform: translate(-50%, -150%); }
                            }
                            @keyframes moveBackward {
                                from { transform: translate(-50%, -50%); }
                                to { transform: translate(-50%, 50%); }
                            }
                            .wheelchair-container {
                                position: relative;
                                width: 300px;
                                height: 300px;
                                margin: 0; /* Changed from margin: 0 auto to margin: 0 to align left */
                                border: 1px solid #f0f0f0;
                                border-radius: 10px;
                                background-color: #fafafa;
                                overflow: hidden;
                            }
                            .wheelchair {
                                position: absolute;
                                top: 50%;
                                left: 50%;
                                transform: translate(-50%, -50%);
                                font-size: 3em;
                                z-index: 10;
                                color: black;
                            }
                            .arrow {
                                position: absolute;
                                font-size: 0.9em;
                                z-index: 5;
                            }
                            .arrow-label {
                                font-size: 0.7em;
                            }
                            .forward {
                                top: 15%;
                                left: 50%;
                                transform: translateX(-50%);
                            }
                            .backward {
                                bottom: 15%;
                                left: 50%;
                                transform: translateX(-50%);
                            }
                            .left {
                                top: 50%;
                                left: 15%;
                                transform: translateY(-50%);
                            }
                            .right {
                                top: 50%;
                                right: 15%;
                                transform: translateY(-50%);
                            }
                            .move-left {
                                animation: moveLeft 2s forwards;
                            }
                            .move-right {
                                animation: moveRight 2s forwards;
                            }
                            .move-forward {
                                animation: moveForward 2s forwards;
                            }
                            .move-backward {
                                animation: moveBackward 2s forwards;
                            }
                            </style>
                            """

                            # Determine animation class based on prediction
                            animation_class = ""
                            if predicted_label == 0:  # Backward
                                animation_class = "move-backward"
                            elif predicted_label == 1:  # Left
                                animation_class = "move-left"
                            elif predicted_label == 2:  # Right
                                animation_class = "move-right"
                            elif predicted_label == 3:  # Forward
                                animation_class = "move-forward"

                            # Create HTML for the animated wheelchair
                            html = f"""
                            {css}
                            <div class="wheelchair-container">
                                <div class="arrow forward">⬆️<br><span class="arrow-label">Forward</span></div>
                                <div class="arrow backward">⬇️<br><span class="arrow-label">Backward</span></div>
                                <div class="arrow left">⬅️<br><span class="arrow-label">Left</span></div>
                                <div class="arrow right">➡️<br><span class="arrow-label">Right</span></div>
                                <div class="wheelchair {animation_class}">♿</div>
                            </div>
                            """

                            # Display the animated wheelchair
                            st.markdown(html, unsafe_allow_html=True)

                        # Right column for signal visualization - wider to show all values
                        with viz_cols[1]:
                            st.subheader("Signal Visualization")

                            # Create a plot showing all values as in the original
                            fig = go.Figure()

                            # Add main signal trace with all values
                            fig.add_trace(go.Scatter(
                                y=list(input_data.values())[2:],  # All values except timestamps
                                mode='lines+markers',
                                name='EEG Signal',
                                line=dict(color='#3498db', width=2),
                                marker=dict(size=5, color='#2c3e50')
                            ))

                            # Update layout for better appearance while keeping all data points
                            fig.update_layout(
                                title='Input EEG Signal',
                                xaxis_title='Feature',
                                yaxis_title='Value',
                                height=300,
                                margin=dict(l=10, r=10, t=40, b=80),
                                xaxis=dict(
                                    ticktext=list(input_data.keys())[2:],  # Skip timestamps
                                    tickvals=list(range(len(input_data)-2)),
                                    tickangle=45,
                                    tickfont=dict(size=8)
                                )
                            )

                            # Display the plot with left alignment
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Expected {len(feature_names)} values, got {len(values)}")
            except Exception as e:
                st.error(f"Error processing pasted data: {str(e)}")

    with tab4:
        st.header("Single Emotion Prediction")
        
        # Create input container
        input_col = st.container()
        with input_col:
            st.write("Click the below buttons to get emotion predictions.")

            # Example data for each emotion (replace with your real samples)
            angry_sample = "1.49E+01,3.16E+01,-1.43E+02,1.98E+01,2.43E+01,-5.84E-01,-2.84E-01,8.82E+00,2.30E+00,-1.97E+00,1.73E+01,3.20E+01,-1.48E+02,2.04E+01,2.28E+01,1.32E+01,3.15E+01,-1.47E+02,1.69E+01,2.77E+01,1.57E+01,3.07E+01,-1.42E+02,2.07E+01,2.28E+01,1.36E+01,3.22E+01,-1.35E+02,2.12E+01,2.38E+01,4.15E+00,5.56E-01,-3.54E-01,3.46E+00,-4.96E+00,1.63E+00,1.33E+00,-5.83E+00,-2.98E-01,7.77E-04,3.71E+00,-1.76E-01,-1.21E+01,-7.84E-01,-1.07E+00,-2.52E+00,7.71E-01,-5.48E+00,-3.75E+00,4.96E+00,-4.35E-01,-7.33E-01,-1.17E+01,-4.24E+00,3.89E+00,2.08E+00,-1.50E+00,-6.25E+00,-4.86E-01,-1.07E+00,1.97E+01,2.99E+00,2.57E+02,9.10E+00,3.16E+01,-2.07E+00,6.32E-01,1.01E+01,-1.95E+00,-3.85E+00,-2.79E+03,5.49E+00,-1.20E+07,-1.87E+02,4.30E+03,2.71E+05,2.48E+02,9.81E+09,2.26E+04,2.42E+06,-3.18E+06,8.74E+02,-3.92E+12,-1.24E+05,2.58E+07,2.42E+08,1.41E+04,2.44E+15,9.53E+06,8.65E+09,5.22E+01,4.22E+01,1.89E+02,4.29E+01,1.00E+02,-8.21E+00,5.20E+00,2.88E+01,-1.18E+00,-8.39E+00,5.22E+01,3.64E+01,1.49E+02,3.70E+01,9.55E+01,4.30E+01,3.70E+01,1.60E+02,4.29E+01,1.00E+02,4.18E+01,4.22E+01,1.89E+02,3.51E+01,9.17E+01,4.40E+01,3.68E+01,1.52E+02,4.17E+01,7.32E+01,9.20E+00,-5.58E-01,-1.09E+01,-5.93E+00,-4.60E+00,1.04E+01,-5.76E+00,-3.96E+01,1.89E+00,3.78E+00,8.21E+00,-3.33E-01,-2.43E+00,-4.76E+00,2.24E+01,1.15E+00,-5.20E+00,-2.88E+01,7.82E+00,8.39E+00,-9.88E-01,2.24E-01,8.45E+00,1.18E+00,2.70E+01,-2.14E+00,5.42E+00,3.72E+01,-6.65E+00,1.86E+01,-2.35E+01,2.51E+01,-8.19E+02,-6.21E+00,-5.37E+01,2.50E+00,-9.53E-01,-5.32E+00,1.02E+01,-1.68E+01,-2.35E+01,2.67E+01,-6.71E+02,-4.20E+00,-2.74E+01,-2.11E+01,2.60E+01,-8.13E+02,-6.21E+00,-3.69E+01,-1.37E+01,2.51E+01,-8.19E+02,4.09E+00,-3.69E+01,-2.10E+01,2.59E+01,-7.12E+02,4.03E+00,-5.37E+01,-2.46E+00,7.09E-01,1.42E+02,2.01E+00,9.50E+00,-9.84E+00,1.66E+00,1.48E+02,-8.29E+00,9.55E+00,-2.50E+00,8.66E-01,4.05E+01,-8.23E+00,2.63E+01,-7.38E+00,9.53E-01,5.32E+00,-1.03E+01,4.64E-02,-4.11E-02,1.57E-01,-1.02E+02,-1.02E+01,1.68E+01,7.34E+00,-7.96E-01,-1.07E+02,5.74E-02,1.68E+01,4.43E+03,-8.48E+02,-5.90E+01,6.82E+02,9.38E+02,1.71E+02,-9.87E+01,-4.17E+02,-3.48E+03,-1.98E+03,9.25E+02,-6.07E+02,-8.48E+02,3.65E+03,4.15E+01,2.42E+02,-7.83E+02,2.37E+02,1.08E+02,3.30E+02,1.62E+04,-3.36E+02,5.15E+02,-1.84E+02,-5.90E+01,4.15E+01,1.76E+01,-5.04E+01,-1.87E+02,2.30E+01,1.55E+01,1.81E+01,2.34E+01,1.34E+02,3.21E+01,-1.26E+01,6.82E+02,2.42E+02,-5.04E+01,2.05E+03,6.04E+02,-7.32E+02,-3.75E+01,-3.99E+02,-6.51E+03,1.72E+03,-3.72E+02,4.61E+01,9.38E+02,-7.83E+02,-1.87E+02,6.04E+02,2.70E+03,4.60E+01,-7.64E+01,-4.96E+02,-4.20E+03,-4.12E+03,-1.61E+02,5.47E+02,1.71E+02,2.37E+02,2.30E+01,-7.32E+02,4.60E+01,2.54E+03,-1.98E+01,-2.04E+02,2.55E+03,-3.81E+03,1.60E+03,6.68E+01,-9.87E+01,1.08E+02,1.55E+01,-3.75E+01,-7.64E+01,-1.98E+01,1.44E+02,-3.41E+01,-2.28E+02,3.14E+01,1.19E+02,3.70E+02,-4.17E+02,3.30E+02,1.81E+01,-3.99E+02,-4.96E+02,-2.04E+02,-3.41E+01,2.89E+02,3.58E+03,2.93E+02,-1.77E+02,-2.22E+02,-3.48E+03,1.62E+04,2.34E+01,-6.51E+03,-4.20E+03,2.55E+03,-2.28E+02,3.58E+03,1.20E+05,-4.98E+03,3.35E+03,-2.50E+03,-1.98E+03,-3.36E+02,1.34E+02,1.72E+03,-4.12E+03,-3.81E+03,3.14E+01,2.93E+02,-4.98E+03,4.52E+04,4.14E+02,-9.43E+02,9.25E+02,5.15E+02,3.21E+01,-3.72E+02,-1.61E+02,1.60E+03,1.19E+02,-1.77E+02,3.35E+03,4.14E+02,1.90E+03,4.51E+02,-6.07E+02,-1.84E+02,-1.26E+01,4.61E+01,5.47E+02,6.68E+01,3.70E+02,-2.22E+02,-2.50E+03,-9.43E+02,4.51E+02,1.79E+03,1.24E+05,4.58E+04,5.01E+03,3.74E+03,2.77E+03,2.22E+03,1.34E+03,2.44E+02,9.06E+01,3.04E+01,2.73E-01,1.36E-12,7.76E+00,5.71E+00,-2.32E+01,5.91E+00,7.27E+00,6.51E+00,-7.26E-02,3.53E+00,1.16E+01,1.06E+01,5.03E+00,6.23E+00,-8.76E-01,3.62E+00,-3.67E+00,1.51E-02,-5.15E-02,7.38E-01,-6.48E-01,1.97E-01,-5.23E-02,3.47E-01,1.69E+00,1.13E+00,2.03E+00,-2.70E+00,1.43E-01,-8.51E-01,-7.00E-01,-3.47E-01,-5.51E-02,-5.03E-03,-2.20E-01,8.04E-01,2.89E-02,1.64E-01,-2.14E+00,-8.09E-01,1.04E-01,1.92E-01,9.16E-02,8.14E-03,3.08E-01,-5.58E-01,-6.80E+00,-5.18E-01,-1.25E-01,-4.34E-01,-1.74E+00,-2.12E-01,-6.36E-01,2.41E+00,-3.11E+00,-5.00E-01,-3.02E-01,2.15E+00,2.44E+00,1.09E+00,-9.51E-02,-5.74E-01,2.44E-01,-2.24E-01,-8.57E-01,-2.92E-01,8.57E-01,1.04E-01,-1.04E+00,4.99E-01,8.54E-03,-1.62E-01,-1.47E+00,7.62E-01,-2.33E-01,1.24E+00,2.09E+00,1.38E+00,-8.90E-01,-9.92E-01,0,5.01E+00,0,0,0,1.85E+03,1.14E+05,2.09E+03,2.53E+03,2.72E+05,4.20E+02,-2.50E+03,3.77E+05,3.27E+03,4.49E+03,1.93E+05,-3.64E+03,-2.11E+04,2.25E+05,-1.40E+04,-2.64E+03,1.03E+05,1.04E+04,3.70E+03,1.79E+05,2.13E+03,6.56E+03,3.41E+05,1.88E+04,-7.65E+03,1.88E+05,-5.90E+03,-2.66E+04,3.42E+05,1.54E+03,-1.10E+04,1.63E+05,5.61E+03,-2.26E+04,1.73E+05,-9.54E+03,-2.12E+04,2.08E+05,-5.14E+02,4.26E+03,1.80E+05,1.41E+04,1.49E+04,2.55E+05,1.06E+04,3.15E+03,1.65E+05,6.08E+03,-8.33E+03,3.35E+05,1.61E+04,2.23E+04,9.98E+04,3.45E+03,6.21E+03,2.12E+05,-6.08E+03,7.97E+03,2.80E+05,1.08E+04,4.55E+03,1.75E+05,1.41E+04,2.80E+03,2.10E+05,4.64E+03,5.52E+03,1.25E+05,5.70E+02,-2.11E+04,4.55E+05,8.63E+03,3.99E+03,1.33E+05,5.82E+03,1.19E+02,6.25E+00,1.09E+01,1.09E+01,6.25E+00,-3.33E+02,2.83E+02,-1.44E+02,-1.44E+02,2.83E+02,8.58E+01,3.97E+01,-5.02E+00,-5.02E+00,3.97E+01,2.09E+02,-4.75E+01,1.51E+01,1.51E+01,-4.75E+01,-5.31E+02,4.49E+02,-2.09E+02,-2.09E+02,4.49E+02,2.05E+02,5.05E+01,-3.19E+01,-3.19E+01,5.05E+01,5.76E+01,1.16E+02,-5.39E+01,-5.39E+01,1.16E+02,-5.45E+02,5.08E+02,-2.54E+02,-2.54E+02,5.08E+02,2.39E+02,-4.62E+01,2.17E+01,2.17E+01,-4.62E+01,2.10E+02,-4.72E+01,1.45E+01,1.45E+01,-4.72E+01,-3.64E+02,2.78E+02,-1.33E+02,-1.33E+02,2.78E+02,-9.03E+00,9.30E+01,6.31E+00,6.31E+00,9.30E+01,-1.18E+02,2.02E+02,-8.14E+01,-8.14E+01,2.02E+02,-3.72E+02,3.81E+02,-2.29E+02,-2.29E+02,3.81E+02,3.02E+02,-4.89E+01,2.12E+01,2.12E+01,-4.89E+01,3.17E+02,-7.95E+01,6.54E+00,6.54E+00,-7.95E+01,-1.75E+02,2.42E+02,-1.40E+02,-1.40E+02,2.42E+02,2.69E+02,-2.65E+01,2.24E+01,2.24E+01,-2.65E+01,1.56E+02,4.74E+00,1.50E+01,1.50E+01,4.74E+00,-2.96E+02,3.84E+02,-2.32E+02,-2.32E+02,3.84E+02,3.36E+01,6.62E+01,6.34E+00,6.34E+00,6.62E+01,6.18E+01,2.09E+01,1.33E+01,1.33E+01,2.09E+01,-3.88E+02,3.26E+02,-1.51E+02,-1.51E+02,3.26E+02,2.51E+02,-1.66E+01,-1.58E+01,-1.58E+01,-1.66E+01,2.24E+00,4.59E+01,-7.94E+00,-7.94E+00,4.59E+01,-6.45E+02,5.14E+02,-2.50E+02,-2.50E+02,5.14E+02,-2.54E+01,1.10E+02,-2.28E+01,-2.28E+01,1.10E+02,-1.11E+02,1.20E+02,-4.63E+01,-4.63E+01,1.20E+02,-6.05E+02,4.74E+02,-2.29E+02,-2.29E+02,4.74E+02,6.03E+01,5.60E+01,-9.45E+00,-9.45E+00,5.60E+01,1.02E+02,-4.57E+01,4.24E+01,4.24E+01,-4.57E+01,-2.41E+02,2.43E+02,-1.48E+02,-1.48E+02,2.43E+02,9.48E+01,2.32E+00,6.17E+00,6.17E+00,2.32E+00,-1.42E+02,1.82E+02,-7.86E+01,-7.86E+01,1.82E+02,-3.84E+02,3.84E+02,-2.18E+02,-2.18E+02,3.84E+02,2.00E+02,-6.10E+01,2.95E+01,2.95E+01,-6.10E+01,2.75E+02,-8.98E+01,4.37E+00,4.37E+00,-8.98E+01,-4.05E+02,3.31E+02,-1.51E+02,-1.51E+02,3.31E+02,2.04E+00,4.62E+01,2.79E+01,2.79E+01,4.62E+01,1.55E+02,-3.04E+01,2.31E-01,2.31E-01,-3.04E+01,-5.58E+02,4.37E+02,-2.08E+02,-2.08E+02,4.37E+02,2.10E+02,-8.07E+01,2.52E+01,2.52E+01,-8.07E+01,-4.17E+01,8.76E+01,-3.81E+01,-3.81E+01,8.76E+01,-4.20E+02,3.95E+02,-2.28E+02,-2.28E+02,3.95E+02,2.00E+02,-3.89E+01,8.43E+00,8.43E+00,-3.89E+01,9.73E+01,-1.87E+01,1.72E+01,1.72E+01,-1.87E+01,-4.60E+02,4.11E+02,-1.99E+02,-1.99E+02,4.11E+02,3.94E+01,8.06E+01,-3.95E+01,-3.95E+01,8.06E+01,-2.16E+01,8.79E+01,-3.36E+01,-3.36E+01,8.79E+01,-5.76E+02,4.49E+02,-1.81E+02,-1.81E+02,4.49E+02,3.04E+02,-8.81E+01,1.44E+01,1.44E+01,-8.81E+01,3.03E+02,-8.78E+01,7.51E+00,7.51E+00,-8.78E+01,-3.12E+02,2.54E+02,-1.24E+02,-1.24E+02,2.54E+02,1.02E+02,4.21E+01,5.13E+00,5.13E+00,4.21E+01,1.96E+02,4.16E+01,-7.00E+01,-7.00E+01,4.16E+01,-3.04E+02,3.58E+02,-2.13E+02,-2.13E+02,3.58E+02,2.61E+02,-2.89E+01,-1.41E+01,-1.41E+01,-2.89E+01,2.51E+01,2.12E+01,-3.87E+00,-3.87E+00,2.12E+01,-3.88E+02,3.11E+02,-1.69E+02,-1.69E+02,3.11E+02,1.36E+02,-6.19E+00,2.39E+00,2.39E+00,-6.19E+00,1.23E+02,2.67E+01,-2.62E+01,-2.62E+01,2.67E+01,-4.39E+02,4.11E+02,-2.31E+02,-2.31E+02,4.11E+02,2.76E+02,-2.55E+01,-3.53E+01,-3.53E+01,-2.55E+01,1.88E+02,1.65E+00,-4.17E+01,-4.17E+01,1.65E+00,-2.85E+02,2.65E+02,-1.50E+02,-1.50E+02,2.65E+02,1.23E+02,1.97E+01,1.48E+01,1.48E+01,1.97E+01,2.14E+02,-2.96E+01,2.56E+01,2.56E+01,-2.96E+01,-3.67E+02,3.63E+02,-1.93E+02,-1.93E+02,3.63E+02,1.16E+02,9.76E+01,-4.80E+01,-4.80E+01,9.76E+01,-2.36E+02,2.38E+02,-6.91E+01,-6.91E+01,2.38E+02,-7.61E+02,6.69E+02,-3.25E+02,-3.25E+02,6.69E+02,1.47E+02,3.44E+01,-4.49E+01,-4.49E+01,3.44E+01,2.17E+02,-5.22E+01,-2.10E+00,-2.10E+00,-5.22E+01,-1.71E+02,2.00E+02,-1.52E+02,-1.52E+02,2.00E+02,1.10E+02,6.96E+00,-9.08E+00,-9.08E+00,6.96E+00,5.84E+01,3.87E+01,-2.08E+01,-2.08E+01,3.87E+01,-1.89E+02,2.63E+02,-1.86E+02,-1.86E+02,2.63E+02,2.61E+02,-6.90E+01,2.83E+01,2.83E+01,-6.90E+01,1.10E+02,2.56E+01,-4.61E+01,-4.61E+01,2.56E+01,-3.68E+02,3.89E+02,-2.36E+02,-2.36E+02,3.89E+02,3.74E+00,6.34E+01,8.51E+00,8.51E+00,6.34E+01,5.70E+01,-2.84E+01,4.09E+01,4.09E+01,-2.84E+01,-5.75E+02,4.87E+02,-2.10E+02,-2.10E+02,4.87E+02,9.54E+01,3.73E+01,-1.77E+00,-1.77E+00,3.73E+01,1.40E+02,3.44E+01,-4.79E+01,-4.79E+01,3.44E+01,-4.70E+02,4.18E+02,-2.04E+02,-2.04E+02,4.18E+02,1.61E+02,-4.75E+01,5.06E+01,5.06E+01,-4.75E+01,2.36E+02,-6.37E+01,-1.38E+01,-1.38E+01,-6.37E+01,-3.96E+02,3.82E+02,-1.95E+02,-1.95E+02,3.82E+02,-3.15E+01,1.59E+02,-7.15E+01,-7.15E+01,1.59E+02,3.01E+01,2.34E+01,1.47E+00,1.47E+00,2.34E+01,-2.88E+02,2.74E+02,-1.58E+02,-1.58E+02,2.74E+02,1.49E+02,-3.42E+01,3.55E+01,3.55E+01,-3.42E+01,1.19E+02,-1.74E+01,5.28E+00,5.28E+00,-1.74E+01,-3.75E+02,3.09E+02,-1.45E+02,-1.45E+02,3.09E+02,7.99E+01,-2.40E-01,3.55E+01,3.55E+01,-2.40E-01,-2.02E+02,2.21E+02,-5.44E+01,-5.44E+01,2.21E+02,-7.82E+02,6.46E+02,-2.89E+02,-2.89E+02,6.46E+02,2.71E+02,-5.76E+01,2.69E+01,2.69E+01,-5.76E+01,2.89E+02,-9.27E+01,8.08E+00,8.08E+00,-9.27E+01,-2.22E+02,2.06E+02,-1.20E+02,-1.20E+02,2.06E+02,2.45E+02,-3.16E+01,-5.90E+00,-5.90E+00,-3.16E+01,3.36E+02,-1.14E+02,1.26E+01,1.26E+01,-1.14E+02,-4.91E+02,4.57E+02,-2.25E+02,-2.25E+02,4.57E+02,2.08E+02,-1.24E+01,-9.86E+00,-9.86E+00,-1.24E+01,-4.03E+01,1.43E+02,-5.30E+01,-5.30E+01,1.43E+02,-4.38E+02,4.05E+02,-1.99E+02,-1.99E+02,4.05E+02,3.40E+02,-8.22E+01,-1.54E+00,-1.54E+00,-8.22E+01,1.96E+02,-8.06E+01,3.67E+01,3.67E+01,-8.06E+01,-3.01E+02,2.87E+02,-1.66E+02,-1.66E+02,2.87E+02,-5.80E+01,1.31E+02,1.01E+00,1.01E+00,1.31E+02,-7.72E+01,1.45E+02,-3.50E+01,-3.50E+01,1.45E+02,-3.74E+02,3.76E+02,-2.07E+02,-2.07E+02,3.76E+02,1.68E+02,-4.79E+00,1.77E+01,1.77E+01,-4.79E+00,1.35E+02,-1.26E+01,1.95E+01,1.95E+01,-1.26E+01,-2.33E+02,2.54E+02,-1.44E+02,-1.44E+02,2.54E+02,2.27E+02,-4.26E+01,4.18E+00,4.18E+00,-4.26E+01,-3.36E+01,1.26E+02,-6.51E+01,-6.51E+01,1.26E+02,-4.47E+02,4.25E+02,-2.23E+02,-2.23E+02,4.25E+02,1.60E+02,-4.27E+01,2.49E+01,2.49E+01,-4.27E+01,1.26E+02,-1.08E+01,1.53E+00,1.53E+00,-1.08E+01,-2.65E+02,2.48E+02,-1.43E+02,-1.43E+02,2.48E+02,6.32E+01,7.77E+01,-3.09E+01,-3.09E+01,7.77E+01,1.61E+02,-6.56E+01,5.32E+01,5.32E+01,-6.56E+01,-5.41E+02,4.65E+02,-2.31E+02,-2.31E+02,4.65E+02,2.26E+02,2.44E+00,-1.94E+01,-1.94E+01,2.44E+00,2.44E+02,-5.48E+01,-5.32E+00,-5.32E+00,-5.48E+01,-1.57E+02,2.44E+02,-1.69E+02,-1.69E+02,2.44E+02,1.05E+02,6.24E+00,5.14E+01,5.14E+01,6.24E+00,2.11E+02,-5.68E+01,1.34E+01,1.34E+01,-5.68E+01,-4.38E+02,3.94E+02,-2.09E+02,-2.09E+02,3.94E+02,1.37E+01,1.14E+02,-5.10E+01,-5.10E+01,1.14E+02,-2.50E+02,2.53E+02,-8.61E+01,-8.61E+01,2.53E+02,-6.59E+02,5.69E+02,-2.62E+02,-2.62E+02,5.69E+02,2.84E+02,-7.47E+01,1.53E+01,1.53E+01,-7.47E+01,2.52E+02,-1.10E+02,1.85E+01,1.85E+01,-1.10E+02,-3.19E+02,2.49E+02,-1.35E+02,-1.35E+02,2.49E+02,2.00E+02,-6.30E+01,2.19E+01,2.19E+01,-6.30E+01,1.00E+01,8.69E+01,-9.86E+01,-9.86E+01,8.69E+01,-5.86E+02,4.78E+02,-2.37E+02,-2.37E+02,4.78E+02,2.44E+02,-8.93E+01,2.17E+01,2.17E+01,-8.93E+01,1.09E+02,-2.75E+01,2.64E+01,2.64E+01,-2.75E+01,-1.96E+02,2.10E+02,-1.45E+02,-1.45E+02,2.10E+02,4.37E+01,4.95E+00,5.66E+01,5.66E+01,4.95E+00,1.59E+02,-5.55E+01,2.82E+01,2.82E+01,-5.55E+01,-5.18E+02,4.38E+02,-2.01E+02,-2.01E+02,4.38E+02,2.48E+02,-1.65E+01,-1.80E+01,-1.80E+01,-1.65E+01,1.36E+02,4.03E+01,-5.50E+01,-5.50E+01,4.03E+01,-4.74E+02,4.39E+02,-2.21E+02,-2.21E+02,4.39E+02,2.72E+02,-8.77E+01,3.24E+01,3.24E+01,-8.77E+01,1.49E+01,3.09E+01,-1.11E+02,2.06E+01,2.55E+01,5.84E-01,-1.12E+00,5.55E+01,-7.39E-01,4.62E+00,1.57E+01,3.07E+01,-1.41E+02,2.08E+01,2.23E+01,1.36E+01,3.22E+01,-1.36E+02,2.12E+01,2.40E+01,1.41E+01,2.99E+01,-1.21E+02,1.96E+01,2.02E+01,1.63E+01,3.08E+01,-4.57E+01,2.09E+01,3.52E+01,2.13E+00,-1.54E+00,-5.33E+00,-4.36E-01,-1.74E+00,1.61E+00,7.91E-01,-2.00E+01,1.17E+00,2.13E+00,-5.89E-01,-1.03E-01,-9.53E+01,-1.19E-01,-1.29E+01,-5.22E-01,2.33E+00,-1.47E+01,1.61E+00,3.87E+00,-2.72E+00,1.44E+00,-9.00E+01,3.17E-01,-1.12E+01,-2.20E+00,-8.93E-01,-7.53E+01,-1.29E+00,-1.51E+01,1.97E+01,3.35E+00,2.53E+02,8.19E+00,3.21E+01,1.95E+00,1.39E-01,-2.21E+01,4.36E-01,3.92E+00,-2.86E+03,2.05E+00,-1.09E+07,-1.66E+01,6.17E+03,2.70E+05,3.37E+02,9.20E+09,1.25E+04,3.36E+06,-3.52E+06,5.34E+02,-3.45E+12,5.62E+03,8.15E+07,2.38E+08,1.67E+04,2.22E+15,3.36E+06,1.88E+10,4.85E+01,4.06E+01,3.05E+02,4.17E+01,1.26E+02,4.64E+00,-3.58E+00,1.15E+02,-4.82E+00,3.67E+01,4.13E+01,4.06E+01,1.90E+02,3.51E+01,8.98E+01,4.38E+01,3.66E+01,1.58E+02,4.17E+01,7.23E+01,4.69E+01,3.70E+01,1.89E+02,3.69E+01,7.35E+01,4.85E+01,3.68E+01,3.05E+02,3.65E+01,1.26E+02,-2.50E+00,3.99E+00,3.26E+01,-6.64E+00,1.75E+01,-5.60E+00,3.58E+00,1.35E+00,-1.82E+00,1.63E+01,-7.14E+00,3.79E+00,-1.15E+02,-1.44E+00,-3.67E+01,-3.10E+00,-4.06E-01,-3.13E+01,4.82E+00,-1.19E+00,-4.64E+00,-1.93E-01,-1.48E+02,5.20E+00,-5.41E+01,-1.54E+00,2.13E-01,-1.16E+02,3.81E-01,-5.30E+01,-2.45E+01,2.35E+01,-8.20E+02,-6.10E-01,-5.12E+01,-3.40E+00,-1.53E+00,2.36E+02,-4.72E+00,6.20E+00,-1.32E+01,2.50E+01,-8.20E+02,4.36E+00,-4.19E+01,-2.11E+01,2.58E+01,-7.18E+02,4.11E+00,-5.12E+01,-2.21E+01,2.36E+01,-5.84E+02,-6.10E-01,-4.50E+01,-2.45E+01,2.35E+01,-5.52E+02,4.05E+00,-3.33E+01,7.95E+00,-8.04E-01,-1.02E+02,2.45E-01,9.31E+00,8.89E+00,1.48E+00,-2.36E+02,4.97E+00,3.11E+00,1.14E+01,1.53E+00,-2.68E+02,3.06E-01,-8.65E+00,9.38E-01,2.29E+00,-1.34E+02,4.72E+00,-6.20E+00,3.40E+00,2.33E+00,-1.66E+02,6.09E-02,-1.80E+01,2.46E+00,4.48E-02,-3.24E+01,-4.66E+00,-1.18E+01,3.96E+03,-7.32E+02,-2.93E+02,2.69E+02,9.50E+02,1.10E+03,-2.97E+02,3.01E+03,-2.46E+03,-1.91E+03,-2.10E+03,-6.67E+02,-7.32E+02,1.85E+03,-5.30E+01,-2.72E+02,-3.99E+02,5.33E+02,-3.90E+02,-7.95E+02,1.16E+04,-3.40E+02,-1.04E+03,-5.86E+02,-2.93E+02,-5.30E+01,7.59E+02,-3.89E+02,-1.17E+03,1.77E+02,-1.82E+02,-3.55E+02,-1.19E+03,1.21E+03,-5.36E+02,-2.32E+02,2.69E+02,-2.72E+02,-3.89E+02,4.14E+03,5.24E+02,-1.38E+03,5.44E+02,2.17E+02,-6.94E+03,6.86E+02,3.08E+03,3.20E+02,9.50E+02,-3.99E+02,-1.17E+03,5.24E+02,2.54E+03,2.60E+02,5.29E+02,1.17E+03,-1.53E+03,-5.47E+03,1.43E+03,6.66E+02,1.10E+03,5.33E+02,1.77E+02,-1.38E+03,2.60E+02,8.02E+03,1.12E+03,1.15E+03,5.70E+03,-4.32E+03,-4.74E+03,8.19E+02,-2.97E+02,-3.90E+02,-1.82E+02,5.44E+02,5.29E+02,1.12E+03,1.22E+03,3.66E+02,-3.22E+03,-1.22E+03,-3.76E+02,1.58E+03,3.01E+03,-7.95E+02,-3.55E+02,2.17E+02,1.17E+03,1.15E+03,3.66E+02,2.83E+03,-4.00E+03,-2.48E+03,-2.13E+03,2.64E+02,-2.46E+03,1.16E+04,-1.19E+03,-6.94E+03,-1.53E+03,5.70E+03,-3.22E+03,-4.00E+03,1.03E+05,-5.52E+03,-1.08E+04,-4.37E+03,-1.91E+03,-3.40E+02,1.21E+03,6.86E+02,-5.47E+03,-4.32E+03,-1.22E+03,-2.48E+03,-5.52E+03,2.53E+04,-3.15E+03,-1.41E+03,-2.10E+03,-1.04E+03,-5.36E+02,3.08E+03,1.43E+03,-4.74E+03,-3.76E+02,-2.13E+03,-1.08E+04,-3.15E+03,1.03E+04,-5.20E+02,-6.67E+02,-5.86E+02,-2.32E+02,3.20E+02,6.66E+02,8.19E+02,1.58E+03,2.64E+02,-4.37E+03,-1.41E+03,-5.20E+02,2.26E+03,1.07E+05,2.84E+04,1.49E+04,5.94E+03,4.02E+03,3.11E+03,1.59E+03,6.62E+02,1.54E+02,1.14E+02,6.07E-13,7.85E-01,6.07E+00,5.79E+00,1.51E+00,7.79E+00,4.71E+00,8.24E+00,-1.28E+01,2.33E+00,1.14E+01,9.74E+00,7.99E+00,-9.93E-01,-3.03E-01,-5.47E-01,-3.58E-01,-5.99E-02,-2.75E-01,3.00E+00,8.38E+00,-3.89E-01,-1.86E-01,-6.82E-01,-2.13E+00,6.17E-01,5.01E-01,-3.93E+00,-1.80E-01,1.55E+00,-1.25E+00,2.21E-01,-8.83E-01,-3.89E-01,-1.03E+00,5.53E-01,-4.04E-01,-5.29E-01,1.43E+00,-2.07E-02,4.89E-02,1.07E+00,-2.00E+00,-4.34E-01,3.99E-01,-1.60E-01,3.86E+00,-2.10E-01,-1.17E-01,-5.78E-01,2.30E+00,-5.27E+00,6.96E-01,7.83E-01,-1.80E+00,-2.24E-01,-1.05E+00,-1.05E+00,1.32E+01,-4.58E+00,-6.26E-01,-2.94E-01,1.35E-01,6.53E-02,-1.69E+00,3.42E+00,5.76E-01,-7.65E-01,5.67E-01,-7.65E-01,-1.35E-03,-2.25E-01,-8.44E-01,-4.79E-01,5.61E-02,-4.75E-01,-2.94E+00,6.16E-02,-9.63E-01,1.97E+00,0,5.00E+00,0,0,0,3.28E+03,1.35E+05,-1.59E+04,5.99E+03,2.52E+05,-1.55E+03,1.13E+03,2.04E+05,2.32E+03,-1.58E+01,2.12E+05,9.85E+03,5.60E+02,1.60E+05,2.44E+03,2.72E+03,2.06E+05,5.72E+03,2.83E+03,1.83E+05,3.96E+03,-2.93E+04,3.45E+05,-9.33E+03,-2.08E+04,1.51E+05,1.00E+04,1.99E+04,1.29E+05,8.05E+03,-7.06E+03,2.48E+05,3.02E+04,-5.95E+03,1.44E+05,-6.65E+03,-2.86E+04,1.95E+05,2.94E+03,4.14E+03,1.51E+05,2.24E+04,-2.11E+04,1.22E+05,2.52E+03,7.54E+03,8.24E+04,-4.21E+03,1.67E+04,1.81E+05,2.79E+04,1.20E+04,1.73E+05,6.50E+03,1.01E+04,9.38E+04,-8.58E+03,-3.08E+04,3.00E+05,2.29E+04,1.21E+04,1.69E+05,1.84E+04,-8.32E+03,1.77E+05,9.02E+03,3.26E+03,8.87E+04,-6.78E+03,4.42E+03,2.21E+05,2.29E+04,3.27E+03,1.80E+05,9.22E+03,6.95E+01,3.89E+00,1.21E+01,1.21E+01,3.89E+00,-1.88E+02,2.74E+02,-2.00E+02,-2.00E+02,2.74E+02,2.58E+02,-7.48E+01,3.78E+01,3.78E+01,-7.48E+01,1.13E+02,2.93E+01,-5.37E+01,-5.37E+01,2.93E+01,-3.71E+02,3.87E+02,-2.29E+02,-2.29E+02,3.87E+02,7.29E+00,6.53E+01,2.76E+00,2.76E+00,6.53E+01,5.35E+01,-2.99E+01,4.62E+01,4.62E+01,-2.99E+01,-5.71E+02,4.88E+02,-2.15E+02,-2.15E+02,4.88E+02,9.20E+01,3.64E+01,2.73E+00,2.73E+00,3.64E+01,1.43E+02,3.52E+01,-5.22E+01,-5.22E+01,3.52E+01,-4.73E+02,4.17E+02,-2.00E+02,-2.00E+02,4.17E+02,1.65E+02,-4.71E+01,4.69E+01,4.69E+01,-4.71E+01,2.33E+02,-6.40E+01,-1.02E+01,-1.02E+01,-6.40E+01,-3.92E+02,3.82E+02,-1.99E+02,-1.99E+02,3.82E+02,-3.47E+01,1.59E+02,-6.82E+01,-6.82E+01,1.59E+02,3.33E+01,2.34E+01,-1.66E+00,-1.66E+00,2.34E+01,-2.91E+02,2.74E+02,-1.55E+02,-1.55E+02,2.74E+02,1.52E+02,-3.44E+01,3.26E+01,3.26E+01,-3.44E+01,1.16E+02,-1.72E+01,8.03E+00,8.03E+00,-1.72E+01,-3.71E+02,3.08E+02,-1.48E+02,-1.48E+02,3.08E+02,7.67E+01,1.97E-01,3.80E+01,3.80E+01,1.97E-01,-1.99E+02,2.21E+02,-5.68E+01,-5.68E+01,2.21E+02,-7.85E+02,6.47E+02,-2.87E+02,-2.87E+02,6.47E+02,2.74E+02,-5.83E+01,2.47E+01,2.47E+01,-5.83E+01,2.86E+02,-9.20E+01,1.02E+01,1.02E+01,-9.20E+01,-2.18E+02,2.06E+02,-1.22E+02,-1.22E+02,2.06E+02,2.42E+02,-3.07E+01,-3.98E+00,-3.98E+00,-3.07E+01,3.40E+02,-1.15E+02,1.07E+01,1.07E+01,-1.15E+02,-4.95E+02,4.58E+02,-2.23E+02,-2.23E+02,4.58E+02,2.11E+02,-1.35E+01,-1.15E+01,-1.15E+01,-1.35E+01,-4.38E+01,1.44E+02,-5.15E+01,-5.15E+01,1.44E+02,-4.35E+02,4.04E+02,-2.00E+02,-2.00E+02,4.04E+02,3.37E+02,-8.07E+01,-1.79E-01,-1.79E-01,-8.07E+01,1.99E+02,-8.21E+01,3.54E+01,3.54E+01,-8.21E+01,-3.05E+02,2.89E+02,-1.65E+02,-1.65E+02,2.89E+02,-5.42E+01,1.29E+02,-7.81E-02,-7.81E-02,1.29E+02,-8.10E+01,1.47E+02,-3.40E+01,-3.40E+01,1.47E+02,-3.70E+02,3.74E+02,-2.08E+02,-2.08E+02,3.74E+02,1.64E+02,-2.80E+00,1.85E+01,1.85E+01,-2.80E+00,1.39E+02,-1.47E+01,1.88E+01,1.88E+01,-1.47E+01,-2.37E+02,2.56E+02,-1.44E+02,-1.44E+02,2.56E+02,2.32E+02,-4.49E+01,3.67E+00,3.67E+00,-4.49E+01,-3.78E+01,1.29E+02,-6.47E+01,-6.47E+01,1.29E+02,-4.43E+02,4.23E+02,-2.23E+02,-2.23E+02,4.23E+02,1.56E+02,-4.01E+01,2.51E+01,2.51E+01,-4.01E+01,1.31E+02,-1.36E+01,1.44E+00,1.44E+00,-1.36E+01,-2.70E+02,2.51E+02,-1.43E+02,-1.43E+02,2.51E+02,6.77E+01,7.47E+01,-3.08E+01,-3.08E+01,7.47E+01,1.56E+02,-6.25E+01,5.30E+01,5.30E+01,-6.25E+01,-5.36E+02,4.61E+02,-2.31E+02,-2.31E+02,4.61E+02,2.22E+02,5.86E+00,-1.99E+01,-1.99E+01,5.86E+00,2.48E+02,-5.84E+01,-4.67E+00,-4.67E+00,-5.84E+01,-1.62E+02,2.48E+02,-1.70E+02,-1.70E+02,2.48E+02,1.10E+02,2.34E+00,5.23E+01,5.23E+01,2.34E+00,2.06E+02,-5.27E+01,1.23E+01,1.23E+01,-5.27E+01,-4.33E+02,3.90E+02,-2.08E+02,-2.08E+02,3.90E+02,8.18E+00,1.19E+02,-5.25E+01,-5.25E+01,1.19E+02,-2.44E+02,2.49E+02,-8.45E+01,-8.45E+01,2.49E+02,-6.65E+02,5.74E+02,-2.64E+02,-2.64E+02,5.74E+02,2.90E+02,-7.99E+01,1.74E+01,1.74E+01,-7.99E+01,2.46E+02,-1.04E+02,1.62E+01,1.62E+01,-1.04E+02,-3.13E+02,2.44E+02,-1.32E+02,-1.32E+02,2.44E+02,1.93E+02,-5.69E+01,1.90E+01,1.90E+01,-5.69E+01,1.68E+01,8.05E+01,-9.53E+01,-9.53E+01,8.05E+01,-5.93E+02,4.85E+02,-2.41E+02,-2.41E+02,4.85E+02,2.51E+02,-9.66E+01,2.57E+01,2.57E+01,-9.66E+01,1.01E+02,-1.97E+01,2.20E+01,2.20E+01,-1.97E+01,-1.88E+02,2.01E+02,-1.40E+02,-1.40E+02,2.01E+02,3.55E+01,1.40E+01,5.09E+01,5.09E+01,1.40E+01,1.68E+02,-6.54E+01,3.46E+01,3.46E+01,-6.54E+01,-5.27E+02,4.49E+02,-2.08E+02,-2.08E+02,4.49E+02,2.58E+02,-2.86E+01,-9.49E+00,-9.49E+00,-2.86E+01,1.26E+02,5.38E+01,-6.51E+01,-6.51E+01,5.38E+01,-4.64E+02,4.23E+02,-2.08E+02,-2.08E+02,4.23E+02,2.62E+02,-6.92E+01,1.66E+01,1.66E+01,-6.92E+01,5.96E+01,-1.89E+01,4.98E+01,4.98E+01,-1.89E+01,-3.63E+02,3.10E+02,-1.73E+02,-1.73E+02,3.10E+02,-5.39E+01,1.11E+02,-2.99E+01,-2.99E+01,1.11E+02,1.27E+01,1.23E+02,-7.71E+01,-7.71E+01,1.23E+02,-4.87E+02,4.11E+02,-1.92E+02,-1.92E+02,4.11E+02,1.66E+02,-3.08E+01,3.41E+01,3.41E+01,-3.08E+01,1.14E+02,3.32E+00,-1.72E+01,-1.72E+01,3.32E+00,-3.04E+02,2.41E+02,-1.27E+02,-1.27E+02,2.41E+02,1.83E+02,-3.10E+01,-2.05E+00,-2.05E+00,-3.10E+01,-1.93E+01,4.56E+01,1.97E-01,1.97E-01,4.56E+01,-3.37E+02,3.27E+02,-2.08E+02,-2.08E+02,3.27E+02,1.73E+02,-5.99E+01,4.21E+01,4.21E+01,-5.99E+01,4.44E+01,1.82E+01,-3.18E+01,-3.18E+01,1.82E+01,-2.97E+02,2.67E+02,-1.74E+02,-1.74E+02,2.67E+02,1.29E+02,3.82E+01,-4.73E+01,-4.73E+01,3.82E+01,7.70E+00,5.87E+01,-2.35E+01,-2.35E+01,5.87E+01,-4.96E+02,4.79E+02,-2.69E+02,-2.69E+02,4.79E+02,1.82E+02,9.02E+00,8.31E+00,8.31E+00,9.02E+00,1.68E+02,3.58E+01,-4.10E+01,-4.10E+01,3.58E+01,-3.65E+02,3.91E+02,-2.25E+02,-2.25E+02,3.91E+02,2.52E+02,-6.50E+01,3.59E+01,3.59E+01,-6.50E+01,2.18E+02,-6.75E+01,3.26E+01,3.26E+01,-6.75E+01,-4.12E+02,3.19E+02,-1.13E+02,-1.13E+02,3.19E+02,3.10E+01,1.29E+02,-2.70E+01,-2.70E+01,1.29E+02,-5.60E+01,1.48E+02,-2.79E+01,-2.79E+01,1.48E+02,-4.93E+02,4.29E+02,-1.85E+02,-1.85E+02,4.29E+02,1.69E+02,-6.42E+01,7.99E+01,7.99E+01,-6.42E+01,1.66E+02,-6.68E+01,3.22E+01,3.22E+01,-6.68E+01,-1.06E+02,1.88E+02,-1.60E+02,-1.60E+02,1.88E+02,1.69E+02,-1.88E+01,3.07E+01,3.07E+01,-1.88E+01,1.58E+02,-2.23E+01,-1.41E+01,-1.41E+01,-2.23E+01,-4.12E+02,3.92E+02,-2.17E+02,-2.17E+02,3.92E+02,2.80E+02,-1.04E+02,3.03E+01,3.03E+01,-1.04E+02,1.63E+01,6.98E+01,-4.77E+01,-4.77E+01,6.98E+01,-3.23E+02,3.02E+02,-1.88E+02,-1.88E+02,3.02E+02,1.38E+02,-4.67E+01,6.94E+01,6.94E+01,-4.67E+01,3.10E+02,-1.21E+02,3.42E+01,3.42E+01,-1.21E+02,-3.36E+02,3.40E+02,-1.74E+02,-1.74E+02,3.40E+02,8.93E+01,1.49E+01,8.38E+00,8.38E+00,1.49E+01,1.65E+02,9.81E+00,-1.62E+01,-1.62E+01,9.81E+00,-4.16E+02,3.82E+02,-1.98E+02,-1.98E+02,3.82E+02,2.67E+02,-1.37E+02,5.25E+01,5.25E+01,-1.37E+02,3.18E+02,-1.14E+02,1.49E+01,1.49E+01,-1.14E+02,-2.02E+02,1.71E+02,-9.54E+01,-9.54E+01,1.71E+02,1.23E+02,3.15E+01,1.08E+01,1.08E+01,3.15E+01,2.47E+02,-3.68E+01,7.28E+00,7.28E+00,-3.68E+01,-1.56E+02,2.07E+02,-1.17E+02,-1.17E+02,2.07E+02,3.42E+02,-1.15E+02,6.07E+01,6.07E+01,-1.15E+02,2.57E+02,-9.03E+01,6.21E+01,6.21E+01,-9.03E+01,-2.44E+02,2.50E+02,-1.26E+02,-1.26E+02,2.50E+02,5.26E+02,-1.71E+02,2.99E+01,2.99E+01,-1.71E+02,2.16E+02,2.58E+00,-4.04E+01,-4.04E+01,2.58E+00,-4.19E+02,4.76E+02,-2.81E+02,-2.81E+02,4.76E+02,3.23E+02,-1.33E+02,8.14E+01,8.14E+01,-1.33E+02,1.47E+02,-7.91E+01,6.44E+01,6.44E+01,-7.91E+01,-1.23E+02,1.55E+02,-1.16E+02,-1.16E+02,1.55E+02,2.48E+02,-6.81E+01,1.03E+01,1.03E+01,-6.81E+01,1.87E+02,-8.27E+01,3.17E+01,3.17E+01,-8.27E+01,-3.81E+02,3.13E+02,-1.84E+02,-1.84E+02,3.13E+02,2.41E+02,-4.75E+01,3.65E+00,3.65E+00,-4.75E+01,1.83E+02,-2.03E+01,-2.05E+01,-2.05E+01,-2.03E+01,-5.05E+02,3.73E+02,-1.63E+02,-1.63E+02,3.73E+02,2.58E+02,-1.13E+02,6.60E+01,6.60E+01,-1.13E+02,2.65E+02,-9.87E+01,4.00E+01,4.00E+01,-9.87E+01,-2.36E+02,2.15E+02,-1.06E+02,-1.06E+02,2.15E+02,1.85E+02,1.36E+01,-2.09E+01,-2.09E+01,1.36E+01,1.92E+02,-1.21E+01,-2.85E+01,-2.85E+01,-1.21E+01,-2.73E+02,2.49E+02,-1.46E+02,-1.46E+02,2.49E+02,3.59E+02,-1.46E+02,1.37E+01,1.37E+01,-1.46E+02,1.19E+02,-7.64E+00,-7.17E+00,-7.17E+00,-7.64E+00,-2.96E+02,3.16E+02,-2.18E+02,-2.18E+02,3.16E+02,3.22E+02,-1.13E+02,3.84E+01,3.84E+01,-1.13E+02,2.45E+02,-6.19E+01,-5.08E+00,-5.08E+00,-6.19E+01,-1.83E+02,2.99E+02,-2.43E+02,-2.43E+02,2.99E+02,1.32E+02,-1.24E+01,9.53E+00,9.53E+00,-1.24E+01"  # Replace ... with real values
            neutral_sample = "2.88E+01,3.31E+01,3.20E+01,2.58E+01,2.28E+01,6.55E+00,1.68E+00,2.88E+00,3.83E+00,-4.82E+00,2.56E+01,3.28E+01,2.96E+01,2.15E+01,1.74E+01,2.55E+01,3.17E+01,3.15E+01,2.62E+01,3.29E+01,3.18E+01,3.31E+01,3.32E+01,2.85E+01,2.68E+01,3.24E+01,3.47E+01,3.38E+01,2.70E+01,1.42E+01,3.42E-02,1.10E+00,-1.87E+00,-4.69E+00,-1.54E+01,-6.22E+00,-3.28E-01,-3.53E+00,-6.98E+00,-9.37E+00,-6.85E+00,-1.89E+00,-4.13E+00,-5.46E+00,3.20E+00,-6.25E+00,-1.43E+00,-1.65E+00,-2.28E+00,6.07E+00,-6.88E+00,-3.00E+00,-2.26E+00,-7.62E-01,1.86E+01,-6.30E-01,-1.56E+00,-6.05E-01,1.52E+00,1.26E+01,7.10E+00,3.55E+00,1.03E+01,6.51E+00,3.31E+01,-5.25E-01,-5.16E-01,-1.91E+00,-1.67E+00,3.15E-01,2.18E+01,-1.82E+01,-3.44E+02,-2.97E+01,-1.04E+04,6.69E+03,4.73E+02,3.18E+04,5.48E+03,3.52E+06,6.92E+03,-1.86E+03,-3.11E+05,-9.12E+03,-8.92E+07,1.19E+06,2.98E+04,1.58E+07,1.07E+06,1.78E+10,4.54E+01,4.16E+01,5.26E+01,4.28E+01,9.63E+01,1.15E+00,-2.14E+00,1.62E+00,-2.31E+00,-6.09E+00,4.42E+01,4.00E+01,4.81E+01,4.28E+01,6.35E+01,3.48E+01,4.16E+01,5.09E+01,3.71E+01,9.63E+01,4.38E+01,3.90E+01,4.94E+01,4.05E+01,9.02E+01,4.54E+01,3.94E+01,5.26E+01,3.76E+01,8.97E+01,9.45E+00,-1.55E+00,-2.82E+00,5.74E+00,-3.28E+01,4.09E-01,1.07E+00,-1.27E+00,2.31E+00,-2.67E+01,-1.15E+00,5.93E-01,-4.44E+00,5.21E+00,-2.62E+01,-9.04E+00,2.62E+00,1.55E+00,-3.43E+00,6.09E+00,-1.06E+01,2.14E+00,-1.62E+00,-5.26E-01,6.53E+00,-1.56E+00,-4.80E-01,-3.17E+00,2.90E+00,4.42E-01,1.36E+01,2.26E+01,-7.05E-01,7.49E+00,-7.02E+01,7.19E+00,3.80E+00,1.27E+01,5.61E+00,-3.55E+00,1.36E+01,2.51E+01,1.11E+01,7.49E+00,-6.67E+01,1.38E+01,2.26E+01,-7.05E-01,1.49E+01,-2.45E+01,2.08E+01,2.64E+01,1.63E+01,1.31E+01,-7.02E+01,2.09E+01,2.79E+01,1.20E+01,1.73E+01,-5.25E+01,-1.82E-01,2.55E+00,1.18E+01,-7.44E+00,-4.21E+01,-7.19E+00,-1.25E+00,-5.14E+00,-5.61E+00,3.55E+00,-7.31E+00,-2.78E+00,-8.44E-01,-9.80E+00,-1.41E+01,-7.01E+00,-3.80E+00,-1.70E+01,1.83E+00,4.57E+01,-7.13E+00,-5.33E+00,-1.27E+01,-2.36E+00,2.80E+01,-1.20E-01,-1.53E+00,4.30E+00,-4.19E+00,-1.77E+01,2.57E+01,-4.59E+01,6.85E+00,4.45E+01,-8.30E+00,-4.11E+01,-1.78E+01,1.14E+01,-2.31E+01,-2.85E+00,-8.68E+00,3.52E+01,-4.59E+01,2.51E+02,-3.44E+01,-3.34E+02,-5.69E+00,3.00E+02,-4.94E+01,-9.04E+01,-9.37E+01,2.29E-01,6.18E+01,7.18E+01,6.85E+00,-3.44E+01,2.18E+01,4.08E+01,-1.09E+00,-1.11E+01,-3.46E+00,1.60E+01,-3.85E+01,6.48E+01,-3.83E+00,-1.28E+01,4.45E+01,-3.34E+02,4.08E+01,5.09E+02,8.10E+01,-3.51E+02,1.01E+02,1.28E+02,2.38E+02,-7.39E+01,-6.70E+01,-1.19E+02,-8.30E+00,-5.69E+00,-1.09E+00,8.10E+01,4.17E+02,-1.10E+02,8.79E+01,5.09E+00,5.87E+01,-5.03E+02,3.49E+01,-8.51E+01,-4.11E+01,3.00E+02,-1.11E+01,-3.51E+02,-1.10E+02,1.07E+03,-5.09E+01,-9.26E+01,-1.79E+02,1.57E+02,1.20E+02,1.03E+02,-1.78E+01,-4.94E+01,-3.46E+00,1.01E+02,8.79E+01,-5.09E+01,1.24E+02,1.66E+01,-3.86E-01,-1.25E+02,-1.62E+01,-1.69E+02,1.14E+01,-9.04E+01,1.60E+01,1.28E+02,5.09E+00,-9.26E+01,1.66E+01,8.29E+01,6.29E+01,3.99E+01,-3.00E+01,-5.30E+01,-2.31E+01,-9.37E+01,-3.85E+01,2.38E+02,5.87E+01,-1.79E+02,-3.86E-01,6.29E+01,6.79E+02,-2.08E+02,-4.32E-01,5.01E+01,-2.85E+00,2.29E-01,6.48E+01,-7.39E+01,-5.03E+02,1.57E+02,-1.25E+02,3.99E+01,-2.08E+02,9.77E+02,-4.45E+01,8.44E+01,-8.68E+00,6.18E+01,-3.83E+00,-6.70E+01,3.49E+01,1.20E+02,-1.62E+01,-3.00E+01,-4.32E-01,-4.45E+01,3.27E+01,3.08E+01,3.52E+01,7.18E+01,-1.28E+01,-1.19E+02,-8.51E+01,1.03E+02,-1.69E+02,-5.30E+01,5.01E+01,8.44E+01,3.08E+01,3.09E+02,1.74E+03,1.16E+03,6.49E+02,4.91E+02,2.53E+02,1.26E+02,4.90E+01,1.91E+01,5.51E+00,3.34E+00,-8.09E-15,4.59E-01,-1.12E+01,-2.90E+00,-1.39E-01,3.70E+00,4.99E+00,6.73E+00,1.35E+00,1.64E+00,6.09E+00,6.32E+00,-5.83E+00,3.34E+00,-9.34E+00,-2.85E+00,-5.26E-01,3.97E-02,-2.26E-01,3.04E-01,-2.23E+00,8.33E-01,-3.95E-01,2.54E-01,-3.43E+00,-2.82E+00,-4.48E+00,5.48E-01,-4.09E-01,7.42E-01,1.87E-01,3.46E-01,5.28E-01,1.29E+00,3.34E-01,-3.32E+00,-2.44E-01,2.55E-02,1.98E+00,7.64E-01,-2.10E-01,-3.35E-02,-4.43E+00,7.71E-01,-1.34E+00,2.50E-01,-1.16E-01,1.78E+00,-2.50E-01,8.43E-02,-3.89E+00,-2.30E+00,-5.18E-01,2.95E+00,8.52E-01,2.02E-01,-1.22E+00,8.49E-01,-3.08E+00,4.06E+00,2.90E+00,-6.47E-01,-2.36E-02,1.27E+00,3.21E-01,5.27E+00,-8.97E-01,5.86E-01,2.79E+00,3.99E-01,-1.96E+00,-1.69E-01,2.60E+00,1.17E+00,-8.07E-01,7.17E+00,4.17E-01,9.83E+00,3.10E+00,5.14E+00,4.98E+00,5.00E+00,0,4.98E+00,0,3.17E+03,4.19E+03,4.10E+03,5.84E+03,5.79E+03,7.46E+03,5.25E+03,4.34E+03,2.77E+03,2.40E+03,3.25E+03,4.21E+03,2.81E+03,6.69E+03,3.62E+03,3.33E+03,1.83E+03,6.20E+03,4.51E+03,3.34E+03,3.18E+03,3.51E+03,-1.11E+01,3.02E+03,2.23E+03,1.82E+03,5.90E+03,3.44E+03,2.34E+03,5.65E+03,4.05E+03,3.73E+03,5.35E+03,5.13E+03,5.52E+03,1.56E+03,3.51E+03,3.93E+03,3.54E+03,3.67E+03,3.88E+03,4.28E+03,4.25E+03,4.49E+03,3.24E+03,3.56E+03,3.44E+03,3.34E+03,3.39E+03,4.92E+03,3.94E+03,2.70E+03,2.45E+03,5.38E+03,4.95E+03,4.16E+03,4.10E+03,3.75E+03,2.91E+03,3.40E+03,3.15E+03,4.40E+03,5.13E+03,3.46E+03,4.64E+03,3.68E+03,1.70E+03,3.34E+03,3.61E+03,8.93E+03,4.01E+03,6.35E+03,1.12E+04,6.05E+03,5.11E+03,6.62E+01,-4.30E+00,4.36E+01,4.36E+01,-4.30E+00,6.60E+01,-2.90E+01,7.33E+01,7.33E+01,-2.90E+01,1.30E+02,5.51E+00,2.70E+01,2.70E+01,5.51E+00,1.81E+02,-3.69E+00,-5.00E+00,-5.00E+00,-3.69E+00,1.70E+02,7.46E+00,-2.52E+01,-2.52E+01,7.46E+00,1.93E+02,8.37E+00,-1.94E+01,-1.94E+01,8.37E+00,1.89E+02,1.88E+01,-8.48E+00,-8.48E+00,1.88E+01,1.75E+02,2.28E+01,-3.66E+01,-3.66E+01,2.28E+01,1.08E+02,-1.96E+01,1.63E+00,1.63E+00,-1.96E+01,1.24E+02,-3.32E+00,-4.06E-01,-4.06E-01,-3.32E+00,8.74E+01,-2.68E+01,2.03E+01,2.03E+01,-2.68E+01,1.48E+02,-1.08E+01,1.51E+00,1.51E+00,-1.08E+01,9.10E+01,-1.01E+01,3.16E+01,3.16E+01,-1.01E+01,1.60E+02,1.33E+01,-2.68E+01,-2.68E+01,1.33E+01,1.34E+02,2.04E+01,-1.01E+01,-1.01E+01,2.04E+01,1.13E+02,1.50E+01,1.82E+01,1.82E+01,1.50E+01,1.43E+02,3.54E+01,-2.84E+01,-2.84E+01,3.54E+01,1.67E+02,1.75E+01,-4.08E+01,-4.08E+01,1.75E+01,1.12E+02,1.81E+01,-1.57E+01,-1.57E+01,1.81E+01,9.67E+01,1.04E+00,-1.04E+01,-1.04E+01,1.04E+00,8.41E+01,-1.68E+00,-6.28E+00,-6.28E+00,-1.68E+00,9.39E+01,8.94E+00,-8.76E+00,-8.76E+00,8.94E+00,4.45E+01,1.99E+00,1.18E+01,1.18E+01,1.99E+00,9.45E+01,1.20E+01,-1.30E+01,-1.30E+01,1.20E+01,8.13E+01,2.99E+00,6.74E+00,6.74E+00,2.99E+00,8.26E+01,-1.87E+01,2.22E+01,2.22E+01,-1.87E+01,1.35E+02,-1.01E+01,-2.18E+01,-2.18E+01,-1.01E+01,1.54E+02,-8.03E+00,-6.44E+00,-6.44E+00,-8.03E+00,9.52E+01,-8.92E+00,6.74E-01,6.74E-01,-8.92E+00,1.85E+02,3.17E-01,-2.46E+01,-2.46E+01,3.17E-01,1.45E+02,5.79E+00,-2.61E+01,-2.61E+01,5.79E+00,1.31E+02,-1.74E+01,7.63E+00,7.63E+00,-1.74E+01,1.63E+02,-1.55E+00,-1.35E+01,-1.35E+01,-1.55E+00,1.61E+02,5.25E-01,-2.27E+01,-2.27E+01,5.25E-01,1.77E+02,5.70E-01,-1.87E+01,-1.87E+01,5.70E-01,9.07E+01,-2.85E+01,9.38E+01,9.38E+01,-2.85E+01,1.24E+02,-2.63E+01,5.69E+01,5.69E+01,-2.63E+01,1.75E+02,8.62E+00,-2.08E+01,-2.08E+01,8.62E+00,1.67E+02,-1.35E+01,7.48E+00,7.48E+00,-1.35E+01,1.64E+02,-1.23E+01,4.08E+00,4.08E+00,-1.23E+01,1.05E+02,-1.23E+01,4.25E+01,4.25E+01,-1.23E+01,1.32E+02,-1.10E+01,2.34E+01,2.34E+01,-1.10E+01,1.28E+02,-8.66E+00,6.23E+00,6.23E+00,-8.66E+00,1.51E+02,-2.09E+00,-9.92E+00,-9.92E+00,-2.09E+00,9.98E+01,-6.82E+00,2.58E+01,2.58E+01,-6.82E+00,1.14E+02,-1.43E+01,2.12E+01,2.12E+01,-1.43E+01,1.41E+02,1.39E+00,-1.35E+01,-1.35E+01,1.39E+00,9.37E+01,6.14E+00,3.40E+01,3.40E+01,6.14E+00,1.13E+02,1.24E+00,2.60E+01,2.60E+01,1.24E+00,1.61E+02,1.81E+01,-3.13E+01,-3.13E+01,1.81E+01,1.29E+02,1.37E+01,-6.96E+00,-6.96E+00,1.37E+01,1.09E+02,2.87E+01,-1.85E+01,-1.85E+01,2.87E+01,7.82E+01,1.36E+01,1.11E+01,1.11E+01,1.36E+01,1.67E+02,1.50E+01,-5.15E+01,-5.15E+01,1.50E+01,1.42E+02,1.75E+01,-3.35E+01,-3.35E+01,1.75E+01,1.39E+02,1.34E+01,-1.57E+01,-1.57E+01,1.34E+01,1.91E+02,1.11E+01,-2.82E+01,-2.82E+01,1.11E+01,2.15E+02,2.10E+01,-5.81E+01,-5.81E+01,2.10E+01,1.23E+02,8.53E+00,-1.18E+01,-1.18E+01,8.53E+00,1.58E+02,-7.28E+00,-4.48E+00,-4.48E+00,-7.28E+00,1.94E+02,6.04E+00,-1.98E+01,-1.98E+01,6.04E+00,1.38E+02,-1.46E+01,-9.32E+00,-9.32E+00,-1.46E+01,1.15E+02,-2.63E+01,3.47E+01,3.47E+01,-2.63E+01,1.40E+02,-1.25E+01,4.77E+00,4.77E+00,-1.25E+01,1.12E+02,-1.51E+01,1.42E+01,1.42E+01,-1.51E+01,2.13E+02,-1.26E+01,-4.71E+01,-4.71E+01,-1.26E+01,2.00E+02,2.91E+01,-6.70E+01,-6.70E+01,2.91E+01,1.73E+02,-2.01E+01,-1.28E+01,-1.28E+01,-2.01E+01,1.77E+02,-2.42E+01,-8.70E+00,-8.70E+00,-2.42E+01,1.92E+02,-7.08E+00,-4.02E+01,-4.02E+01,-7.08E+00,1.16E+02,-1.90E+01,1.10E+01,1.10E+01,-1.90E+01,1.77E+02,-1.83E+01,-5.98E+00,-5.98E+00,-1.83E+01,1.91E+02,1.43E+01,-5.13E+01,-5.13E+01,1.43E+01,1.51E+02,8.40E-01,-4.16E+01,-4.16E+01,8.40E-01,1.32E+02,-1.72E+01,5.32E+00,5.32E+00,-1.72E+01,1.02E+02,-1.34E+01,2.54E+01,2.54E+01,-1.34E+01,9.46E+01,-1.78E+01,2.57E+01,2.57E+01,-1.78E+01,1.68E+02,-1.25E+01,-5.30E+00,-5.30E+00,-1.25E+01,1.63E+02,-6.52E+00,2.52E+01,2.52E+01,-6.52E+00,1.70E+02,4.47E+00,-4.09E+00,-4.09E+00,4.47E+00,1.95E+02,2.53E+00,5.22E+00,5.22E+00,2.53E+00,1.47E+02,-4.20E+00,3.98E+01,3.98E+01,-4.20E+00,1.50E+02,1.79E+00,3.30E+01,3.30E+01,1.79E+00,6.00E+01,-3.56E+01,8.01E+01,8.01E+01,-3.56E+01,9.39E+01,-2.98E+01,3.49E+01,3.49E+01,-2.98E+01,1.64E+02,-1.71E+01,-7.87E+00,-7.87E+00,-1.71E+01,1.42E+02,3.75E+00,3.83E+00,3.83E+00,3.75E+00,1.70E+02,1.77E+00,-1.32E+01,-1.32E+01,1.77E+00,1.90E+02,1.18E+01,-2.87E+01,-2.87E+01,1.18E+01,1.31E+02,1.36E+01,5.76E+00,5.76E+00,1.36E+01,1.07E+02,5.98E+00,2.44E+01,2.44E+01,5.98E+00,8.63E+01,7.45E+00,3.56E+01,3.56E+01,7.45E+00,1.79E+02,1.17E+01,-5.48E+00,-5.48E+00,1.17E+01,2.01E+02,9.77E+00,-4.65E+00,-4.65E+00,9.77E+00,1.68E+02,7.60E+00,9.25E+00,9.25E+00,7.60E+00,1.76E+02,-9.65E+00,1.54E+01,1.54E+01,-9.65E+00,2.08E+02,1.61E+00,-1.93E+01,-1.93E+01,1.61E+00,2.16E+02,1.21E+01,-4.85E+01,-4.85E+01,1.21E+01,1.63E+02,2.93E+00,-1.40E+01,-1.40E+01,2.93E+00,1.66E+02,5.14E+00,-1.93E+01,-1.93E+01,5.14E+00,1.51E+02,4.89E-01,-2.29E+01,-2.29E+01,4.89E-01,2.06E+02,1.23E+01,-3.05E+01,-3.05E+01,1.23E+01,1.09E+02,-1.23E+01,1.28E+01,1.28E+01,-1.23E+01,1.12E+02,7.93E+00,-9.83E+00,-9.83E+00,7.93E+00,1.49E+02,5.79E+00,1.13E+00,1.13E+00,5.79E+00,1.44E+02,1.21E+01,1.04E+01,1.04E+01,1.21E+01,1.37E+02,7.59E+00,1.50E+00,1.50E+00,7.59E+00,1.61E+02,3.28E+00,-1.41E+01,-1.41E+01,3.28E+00,1.49E+02,2.66E+01,-2.70E+01,-2.70E+01,2.66E+01,1.57E+02,-8.17E+00,1.01E+01,1.01E+01,-8.17E+00,1.95E+02,-3.58E+00,-1.75E+01,-1.75E+01,-3.58E+00,1.94E+02,5.07E+00,-1.96E+01,-1.96E+01,5.07E+00,1.24E+02,4.89E+00,1.67E+01,1.67E+01,4.89E+00,1.05E+02,-2.68E+01,5.90E+01,5.90E+01,-2.68E+01,1.09E+02,-9.98E+00,2.46E+01,2.46E+01,-9.98E+00,9.50E+01,-7.96E+00,2.01E+01,2.01E+01,-7.96E+00,1.23E+02,-1.30E+01,1.98E+01,1.98E+01,-1.30E+01,1.68E+02,9.51E+00,-2.09E+01,-2.09E+01,9.51E+00,1.46E+02,-1.23E+01,-7.52E+00,-7.52E+00,-1.23E+01,1.78E+02,-5.31E+00,-4.53E+00,-4.53E+00,-5.31E+00,1.52E+02,5.17E+00,1.49E+00,1.49E+00,5.17E+00,1.29E+02,1.82E+01,1.04E+01,1.04E+01,1.82E+01,1.08E+02,6.21E+00,3.22E+01,3.22E+01,6.21E+00,1.62E+02,1.87E+01,-1.20E+01,-1.20E+01,1.87E+01,1.40E+02,6.22E+00,-2.36E+01,-2.36E+01,6.22E+00,1.48E+02,1.21E+01,-2.06E+01,-2.06E+01,1.21E+01,1.10E+02,1.14E+01,-8.60E-01,-8.60E-01,1.14E+01,1.45E+02,5.88E+00,-2.69E+00,-2.69E+00,5.88E+00,1.59E+02,1.40E+01,-5.94E+00,-5.94E+00,1.40E+01,1.63E+02,1.57E+01,-1.29E+01,-1.29E+01,1.57E+01,1.43E+02,1.18E+01,-1.12E+01,-1.12E+01,1.18E+01,1.27E+02,-8.95E+00,2.51E+01,2.51E+01,-8.95E+00,1.22E+02,-3.50E+00,2.23E+01,2.23E+01,-3.50E+00,1.17E+02,2.30E+00,1.22E+01,1.22E+01,2.30E+00,1.09E+02,-1.54E+01,6.40E+01,6.40E+01,-1.54E+01,1.08E+02,-1.62E+01,3.18E+01,3.18E+01,-1.62E+01,1.54E+02,9.94E-01,-2.88E+00,-2.88E+00,9.94E-01,1.74E+02,-4.69E-01,2.68E+01,2.68E+01,-4.69E-01,7.60E+01,-2.14E+01,6.66E+01,6.66E+01,-2.14E+01,1.22E+02,-3.21E+01,7.50E+01,7.50E+01,-3.21E+01,1.35E+02,-2.75E+01,3.98E+01,3.98E+01,-2.75E+01,1.26E+02,-1.42E+01,5.24E+01,5.24E+01,-1.42E+01,1.02E+02,-1.58E+00,4.53E+01,4.53E+01,-1.58E+00,1.10E+02,-1.28E+01,4.62E+01,4.62E+01,-1.28E+01,2.12E+02,1.73E+01,-1.83E+01,-1.83E+01,1.73E+01,2.13E+02,7.92E+00,-4.28E+01,-4.28E+01,7.92E+00,1.74E+02,-6.07E+00,2.97E+00,2.97E+00,-6.07E+00,2.40E+02,7.87E+00,-3.93E+01,-3.93E+01,7.87E+00,1.84E+02,1.70E+01,-1.89E+01,-1.89E+01,1.70E+01,1.89E+02,-1.96E+01,-8.37E+00,-8.37E+00,-1.96E+01,2.85E+01,3.25E+01,3.45E+01,2.63E+01,2.30E+01,-7.17E+00,-2.71E+00,2.02E+00,-2.89E+00,4.98E+00,3.18E+01,3.31E+01,3.32E+01,2.85E+01,2.73E+01,3.24E+01,3.47E+01,3.37E+01,2.70E+01,1.40E+01,2.80E+01,3.22E+01,3.55E+01,2.64E+01,2.67E+01,2.20E+01,3.02E+01,3.54E+01,2.33E+01,2.44E+01,-6.07E-01,-1.60E+00,-5.58E-01,1.48E+00,1.33E+01,3.83E+00,8.59E-01,-2.38E+00,2.09E+00,6.12E-01,9.82E+00,2.91E+00,-2.22E+00,5.15E+00,2.90E+00,4.44E+00,2.46E+00,-1.82E+00,6.05E-01,-1.27E+01,1.04E+01,4.51E+00,-1.66E+00,3.67E+00,-1.04E+01,5.99E+00,2.05E+00,1.62E-01,3.07E+00,2.28E+00,7.62E+00,3.85E+00,9.15E+00,6.03E+00,3.34E+01,1.35E+00,7.89E-01,1.71E-02,1.06E+00,8.06E-02,-6.41E+01,-3.68E+00,-2.04E+02,8.95E+01,-2.26E+03,9.22E+03,4.90E+02,1.72E+04,4.04E+03,3.52E+06,-3.78E+04,-2.05E+02,-9.72E+04,2.71E+04,-2.23E+06,2.20E+06,2.16E+04,4.84E+06,7.12E+05,1.61E+10,4.54E+01,4.13E+01,5.29E+01,4.49E+01,1.11E+02,-3.59E+00,1.83E+00,1.37E+00,4.72E+00,1.85E+01,4.39E+01,3.91E+01,4.91E+01,4.01E+01,9.28E+01,4.54E+01,3.94E+01,5.15E+01,3.74E+01,8.29E+01,4.18E+01,4.13E+01,5.29E+01,4.49E+01,1.11E+02,3.93E+01,4.08E+01,5.06E+01,3.42E+01,9.07E+01,-1.46E+00,-3.38E-01,-2.48E+00,2.74E+00,9.88E+00,2.14E+00,-2.17E+00,-3.85E+00,-4.72E+00,-1.85E+01,4.56E+00,-1.73E+00,-1.50E+00,5.95E+00,2.04E+00,3.59E+00,-1.83E+00,-1.37E+00,-7.46E+00,-2.84E+01,6.02E+00,-1.39E+00,9.86E-01,3.22E+00,-7.84E+00,2.42E+00,4.36E-01,2.36E+00,1.07E+01,2.06E+01,7.96E+00,2.43E+01,1.27E+01,1.28E+01,-6.49E+01,-1.26E+01,-2.11E+00,8.10E-01,2.03E+00,2.17E+01,2.06E+01,2.64E+01,1.60E+01,1.28E+01,-6.49E+01,2.09E+01,2.79E+01,1.27E+01,1.75E+01,-5.10E+01,1.60E+01,2.47E+01,1.77E+01,1.58E+01,-3.93E+01,7.96E+00,2.43E+01,1.35E+01,1.48E+01,-4.33E+01,-3.48E-01,-1.52E+00,3.34E+00,-4.74E+00,-1.39E+01,4.54E+00,1.72E+00,-1.73E+00,-2.98E+00,-2.57E+01,1.26E+01,2.11E+00,2.53E+00,-2.03E+00,-2.17E+01,4.89E+00,3.24E+00,-5.06E+00,1.76E+00,-1.17E+01,1.30E+01,3.63E+00,-8.10E-01,2.71E+00,-7.72E+00,8.07E+00,3.94E-01,4.25E+00,9.54E-01,4.00E+00,3.16E+01,4.64E+01,3.40E+00,-6.34E+01,2.18E+00,1.91E+01,-4.08E+00,-2.48E+01,-6.77E+01,-4.52E+00,-3.59E+00,-4.60E+00,4.64E+01,2.12E+02,1.47E+01,-2.77E+02,-1.31E+00,1.40E+02,5.54E+01,-1.26E+02,-1.07E+02,-6.44E+01,3.87E+01,-7.02E+00,3.40E+00,1.47E+01,2.05E+01,-1.02E+01,2.24E+01,-2.39E+01,-6.85E+00,2.97E+00,2.11E+01,-4.00E+01,2.65E+01,-8.50E+00,-6.34E+01,-2.77E+02,-1.02E+01,4.69E+02,1.27E+02,-3.76E+02,-7.64E+01,1.82E+02,1.85E+02,-3.09E+01,1.20E+01,1.37E+01,2.18E+00,-1.31E+00,2.24E+01,1.27E+02,6.11E+02,-2.43E+01,1.64E+01,6.81E+01,1.91E+02,-3.80E+02,3.28E+01,-3.48E+00,1.91E+01,1.40E+02,-2.39E+01,-3.76E+02,-2.43E+01,8.82E+02,-1.02E+01,-9.40E+01,-3.22E+01,1.38E+02,-1.89E+02,-4.67E+00,-4.08E+00,5.54E+01,-6.85E+00,-7.64E+01,1.64E+01,-1.02E+01,1.01E+02,-5.20E+01,-1.03E+02,-2.98E+01,6.10E+00,2.19E+01,-2.48E+01,-1.26E+02,2.97E+00,1.82E+02,6.81E+01,-9.40E+01,-5.20E+01,1.08E+02,7.06E+01,-3.54E+01,-1.29E+01,1.52E+01,-6.77E+01,-1.07E+02,2.11E+01,1.85E+02,1.91E+02,-3.22E+01,-1.03E+02,7.06E+01,9.08E+02,-1.76E+02,1.06E+01,-1.21E+02,-4.52E+00,-6.44E+01,-4.00E+01,-3.09E+01,-3.80E+02,1.38E+02,-2.98E+01,-3.54E+01,-1.76E+02,4.20E+02,-5.98E+01,9.26E+00,-3.59E+00,3.87E+01,2.65E+01,1.20E+01,3.28E+01,-1.89E+02,6.10E+00,-1.29E+01,1.06E+01,-5.98E+01,1.17E+02,-9.50E+00,-4.60E+00,-7.02E+00,-8.50E+00,1.37E+01,-3.48E+00,-4.67E+00,2.19E+01,1.52E+01,-1.21E+02,9.26E+00,-9.50E+00,4.19E+01,1.45E+03,9.97E+02,7.39E+02,4.48E+02,9.39E+01,9.71E+01,5.11E+01,2.16E+01,1.73E+01,7.02E+00,-1.43E-14,5.07E-01,-1.20E+01,-3.08E+00,1.88E+00,4.33E+00,5.36E+00,5.47E+00,3.25E+00,-4.85E-01,6.53E+00,4.16E+00,-2.90E-01,8.76E-02,9.47E+00,-1.44E+00,9.23E-02,8.77E-01,9.47E-01,-9.65E-01,-2.90E-02,4.79E-01,-2.79E-02,2.15E+00,-3.19E+00,2.56E+00,-5.64E-01,-5.03E-01,-1.40E+00,7.60E-01,1.39E+00,-4.88E-01,-2.20E+00,-5.81E-01,1.58E+00,-2.33E+00,-1.63E+00,5.70E-01,-9.76E-01,-9.75E-01,-1.90E-01,4.71E-01,3.19E+00,-1.04E+00,3.01E+00,2.32E+00,3.56E-01,1.10E+00,3.70E-01,1.35E+00,-1.68E+00,3.65E+00,-3.77E+00,1.28E+00,-5.52E-01,1.32E-01,-1.79E+00,-2.69E+00,-3.54E-01,-3.42E+00,-5.82E+00,2.08E-01,6.28E-01,1.70E+00,-1.25E+00,5.84E+00,5.32E-01,-9.56E-01,-1.13E+00,1.02E+00,-1.44E+00,-3.17E+00,1.91E+00,-2.13E-01,3.89E+00,5.58E+00,5.73E-01,-8.31E+00,4.42E+00,-5.99E+00,4.97E+00,5.00E+00,4.97E+00,4.98E+00,0,3.64E+03,2.08E+03,2.46E+03,4.73E+03,7.01E+03,5.03E+03,3.96E+03,4.20E+03,1.42E+03,1.35E+03,7.44E+03,6.33E+03,6.67E+03,5.49E+03,4.43E+03,3.12E+03,3.17E+03,7.98E+03,5.58E+03,4.75E+03,5.25E+03,5.45E+03,2.97E+03,4.23E+03,3.06E+03,6.70E+03,8.72E+03,3.12E+03,2.95E+03,4.22E+03,2.74E+03,2.78E+03,2.29E+03,3.63E+03,3.33E+03,6.50E+03,5.06E+03,3.54E+03,2.38E+03,3.26E+03,2.57E+03,3.41E+03,3.72E+03,1.99E+03,3.02E+03,4.10E+03,3.79E+03,3.50E+03,3.40E+03,2.50E+03,5.26E+03,2.67E+03,1.87E+03,4.26E+03,7.09E+03,5.23E+03,3.43E+03,3.09E+03,2.91E+03,3.00E+03,2.99E+03,3.09E+03,4.37E+03,1.76E+03,4.71E+03,4.09E+03,1.84E+03,2.64E+03,3.13E+03,4.37E+03,3.18E+03,8.41E+03,2.69E+03,5.29E+03,6.13E+03,1.26E+02,-4.00E+00,7.42E+00,7.42E+00,-4.00E+00,8.16E+01,-2.23E+01,3.53E+01,3.53E+01,-2.23E+01,1.78E+02,-8.97E+00,-1.28E+01,-1.28E+01,-8.97E+00,1.55E+02,-9.56E+00,3.16E+01,3.16E+01,-9.56E+00,1.77E+02,7.23E+00,-9.78E+00,-9.78E+00,7.23E+00,1.88E+02,-4.05E-02,1.04E+01,1.04E+01,-4.05E-02,1.53E+02,-1.77E+00,3.51E+01,3.51E+01,-1.77E+00,1.44E+02,-5.25E-01,3.73E+01,3.73E+01,-5.25E-01,6.51E+01,-3.34E+01,7.60E+01,7.60E+01,-3.34E+01,8.92E+01,-3.19E+01,3.87E+01,3.87E+01,-3.19E+01,1.68E+02,-1.50E+01,-1.15E+01,-1.15E+01,-1.50E+01,1.38E+02,1.75E+00,7.25E+00,7.25E+00,1.75E+00,1.74E+02,3.70E+00,-1.65E+01,-1.65E+01,3.70E+00,1.87E+02,9.95E+00,-2.57E+01,-2.57E+01,9.95E+00,1.34E+02,1.55E+01,2.81E+00,2.81E+00,1.55E+01,1.04E+02,4.19E+00,2.72E+01,2.72E+01,4.19E+00,8.93E+01,9.19E+00,3.29E+01,3.29E+01,9.19E+00,1.76E+02,1.00E+01,-2.89E+00,-2.89E+00,1.00E+01,2.03E+02,1.14E+01,-7.14E+00,-7.14E+00,1.14E+01,1.66E+02,5.97E+00,1.16E+01,1.16E+01,5.97E+00,1.79E+02,-8.05E+00,1.31E+01,1.31E+01,-8.05E+00,2.06E+02,4.70E-02,-1.71E+01,-1.71E+01,4.70E-02,2.18E+02,1.36E+01,-5.06E+01,-5.06E+01,1.36E+01,1.61E+02,1.43E+00,-1.19E+01,-1.19E+01,1.43E+00,1.68E+02,6.61E+00,-2.13E+01,-2.13E+01,6.61E+00,1.50E+02,-9.55E-01,-2.10E+01,-2.10E+01,-9.55E-01,2.07E+02,1.38E+01,-3.24E+01,-3.24E+01,1.38E+01,1.07E+02,-1.37E+01,1.46E+01,1.46E+01,-1.37E+01,1.13E+02,9.30E+00,-1.15E+01,-1.15E+01,9.30E+00,1.47E+02,4.45E+00,2.78E+00,2.78E+00,4.45E+00,1.45E+02,1.34E+01,8.77E+00,8.77E+00,1.34E+01,1.36E+02,6.30E+00,3.03E+00,3.03E+00,6.30E+00,1.62E+02,4.54E+00,-1.56E+01,-1.56E+01,4.54E+00,1.48E+02,2.54E+01,-2.55E+01,-2.55E+01,2.54E+01,1.58E+02,-6.95E+00,8.75E+00,8.75E+00,-6.95E+00,1.94E+02,-4.78E+00,-1.62E+01,-1.62E+01,-4.78E+00,1.95E+02,6.24E+00,-2.08E+01,-2.08E+01,6.24E+00,1.23E+02,3.74E+00,1.79E+01,1.79E+01,3.74E+00,1.06E+02,-2.56E+01,5.79E+01,5.79E+01,-2.56E+01,1.08E+02,-1.11E+01,2.57E+01,2.57E+01,-1.11E+01,9.53E+01,-6.88E+00,1.91E+01,1.91E+01,-6.88E+00,1.23E+02,-1.41E+01,2.08E+01,2.08E+01,-1.41E+01,1.68E+02,1.06E+01,-2.18E+01,-2.18E+01,1.06E+01,1.46E+02,-1.34E+01,-6.65E+00,-6.65E+00,-1.34E+01,1.77E+02,-4.31E+00,-5.34E+00,-5.34E+00,-4.31E+00,1.52E+02,4.20E+00,2.24E+00,2.24E+00,4.20E+00,1.28E+02,1.91E+01,9.67E+00,9.67E+00,1.91E+01,1.08E+02,5.29E+00,3.29E+01,3.29E+01,5.29E+00,1.62E+02,1.95E+01,-1.25E+01,-1.25E+01,1.95E+01,1.40E+02,5.35E+00,-2.31E+01,-2.31E+01,5.35E+00,1.48E+02,1.30E+01,-2.10E+01,-2.10E+01,1.30E+01,1.11E+02,1.06E+01,-4.74E-01,-4.74E-01,1.06E+01,1.44E+02,6.67E+00,-3.00E+00,-3.00E+00,6.67E+00,1.60E+02,1.33E+01,-5.69E+00,-5.69E+00,1.33E+01,1.62E+02,1.65E+01,-1.30E+01,-1.30E+01,1.65E+01,1.44E+02,1.11E+01,-1.11E+01,-1.11E+01,1.11E+01,1.25E+02,-8.27E+00,2.51E+01,2.51E+01,-8.27E+00,1.23E+02,-4.14E+00,2.22E+01,2.22E+01,-4.14E+00,1.16E+02,2.92E+00,1.24E+01,1.24E+01,2.92E+00,1.11E+02,-1.59E+01,6.37E+01,6.37E+01,-1.59E+01,1.06E+02,-1.57E+01,3.22E+01,3.22E+01,-1.57E+01,1.56E+02,4.90E-01,-3.38E+00,-3.38E+00,4.90E-01,1.72E+02,-6.60E-03,2.74E+01,2.74E+01,-6.60E-03,7.87E+01,-2.19E+01,6.58E+01,6.58E+01,-2.19E+01,1.19E+02,-3.17E+01,7.59E+01,7.59E+01,-3.17E+01,1.38E+02,-2.79E+01,3.87E+01,3.87E+01,-2.79E+01,1.22E+02,-1.39E+01,5.37E+01,5.37E+01,-1.39E+01,1.06E+02,-1.78E+00,4.37E+01,4.37E+01,-1.78E+00,1.06E+02,-1.27E+01,4.81E+01,4.81E+01,-1.27E+01,2.17E+02,1.73E+01,-2.07E+01,-2.07E+01,1.73E+01,2.07E+02,7.84E+00,-3.99E+01,-3.99E+01,7.84E+00,1.81E+02,-5.85E+00,-6.99E-01,-6.99E-01,-5.85E+00,2.31E+02,7.45E+00,-3.45E+01,-3.45E+01,7.45E+00,1.95E+02,1.77E+01,-2.57E+01,-2.57E+01,1.77E+01,1.71E+02,-2.11E+01,2.93E+00,2.93E+00,-2.11E+01,1.39E+02,-2.10E+01,4.96E+00,4.96E+00,-2.10E+01,1.66E+02,-2.24E+01,-1.73E+01,-1.73E+01,-2.24E+01,8.88E+01,-4.03E+01,4.63E+01,4.63E+01,-4.03E+01,1.48E+02,-1.30E+01,5.42E+00,5.42E+00,-1.30E+01,1.89E+02,1.08E+01,-3.37E+01,-3.37E+01,1.08E+01,1.29E+02,-1.47E+01,2.26E+01,2.26E+01,-1.47E+01,8.21E+01,-5.64E+00,3.63E+01,3.63E+01,-5.64E+00,1.60E+02,-1.96E+00,-9.87E+00,-9.87E+00,-1.96E+00,1.62E+02,-1.10E+00,-8.78E+00,-8.78E+00,-1.10E+00,2.42E+02,2.12E+01,-4.60E+01,-4.60E+01,2.12E+01,2.17E+02,2.60E+00,-3.15E+01,-3.15E+01,2.60E+00,2.63E+02,2.20E+01,-5.24E+01,-5.24E+01,2.20E+01,1.90E+02,1.06E+01,-1.30E+00,-1.30E+00,1.06E+01,1.61E+02,4.13E+00,1.72E+01,1.72E+01,4.13E+00,1.58E+02,2.43E+00,9.81E+00,9.81E+00,2.43E+00,1.33E+02,6.77E-01,9.56E-01,9.56E-01,6.77E-01,1.39E+02,-2.22E+01,1.90E+01,1.90E+01,-2.22E+01,2.25E+02,-9.22E-01,-4.14E+01,-4.14E+01,-9.22E-01,1.36E+02,-5.17E+00,-2.29E+01,-2.29E+01,-5.17E+00,1.51E+02,1.96E+01,-4.40E+01,-4.40E+01,1.96E+01,1.49E+02,2.53E+00,-1.92E+01,-1.92E+01,2.53E+00,1.29E+02,2.38E+00,-1.79E+01,-1.79E+01,2.38E+00,1.05E+02,-2.57E+01,1.65E+01,1.65E+01,-2.57E+01,1.38E+02,-7.51E+00,1.97E+01,1.97E+01,-7.51E+00,1.15E+02,-2.10E+01,3.77E+01,3.77E+01,-2.10E+01,2.01E+02,-7.64E+00,-2.84E+01,-2.84E+01,-7.64E+00,1.84E+02,-1.15E+01,-2.33E+01,-2.33E+01,-1.15E+01,1.73E+02,8.93E+00,-3.16E+01,-3.16E+01,8.93E+00,1.23E+02,-1.19E+00,-2.99E+00,-2.99E+00,-1.19E+00,1.42E+02,-2.21E-01,9.61E+00,9.61E+00,-2.21E-01,7.97E+01,-4.67E+00,4.66E+01,4.66E+01,-4.67E+00,7.66E+01,-2.81E+01,4.64E+01,4.64E+01,-2.81E+01,8.06E+01,-1.37E+01,5.02E+01,5.02E+01,-1.37E+01,1.32E+02,-1.98E+00,8.48E+00,8.48E+00,-1.98E+00,9.59E+01,1.00E+01,2.81E+01,2.81E+01,1.00E+01,1.68E+02,1.24E+01,-4.11E+00,-4.11E+00,1.24E+01,1.34E+02,-4.18E+00,-8.82E+00,-8.82E+00,-4.18E+00,1.88E+02,-1.23E+00,-5.54E+01,-5.54E+01,-1.23E+00,1.48E+02,-1.03E+01,-1.76E+01,-1.76E+01,-1.03E+01,1.61E+02,6.24E-01,-2.28E+00,-2.28E+00,6.24E-01,1.32E+02,-1.02E+01,-1.52E+01,-1.52E+01,-1.02E+01,1.42E+02,-1.03E+01,-1.73E+01,-1.73E+01,-1.03E+01,1.14E+02,-5.97E+00,3.37E+00,3.37E+00,-5.97E+00,5.49E+01,-4.59E+01,4.11E+01,4.11E+01,-4.59E+01,9.22E+01,-2.88E+01,3.07E+01,3.07E+01,-2.88E+01,1.34E+02,-7.47E+00,-6.31E+00,-6.31E+00,-7.47E+00,1.58E+02,-1.38E+01,-1.65E-01,-1.65E-01,-1.38E+01,1.51E+02,-1.13E+01,1.22E+01,1.22E+01,-1.13E+01,1.19E+02,3.17E+00,3.56E+01,3.56E+01,3.17E+00,9.44E+01,1.93E+00,7.94E+00,7.94E+00,1.93E+00,1.70E+02,-1.23E+01,-2.75E+01,-2.75E+01,-1.23E+01,1.26E+02,-1.74E+01,-1.31E+01,-1.31E+01,-1.74E+01,5.88E+01,-3.71E+01,3.12E+01,3.12E+01,-3.71E+01,1.36E+02,-2.42E+01,2.41E+00,2.41E+00,-2.42E+01,2.11E+02,1.07E+00,-5.95E+01,-5.95E+01,1.07E+00,1.73E+02,1.67E+01,-3.37E+01,-3.37E+01,1.67E+01,1.25E+02,1.07E+01,-6.68E+00,-6.68E+00,1.07E+01,1.22E+02,1.84E+01,-9.99E+00,-9.99E+00,1.84E+01,1.06E+02,-2.43E+01,1.15E+01,1.15E+01,-2.43E+01,1.55E+02,-1.08E+01,-1.88E+01,-1.88E+01,-1.08E+01,1.38E+02,-6.57E+00,-6.94E+00,-6.94E+00,-6.57E+00,9.53E+01,1.04E+00,2.01E+00,2.01E+00,1.04E+00,1.20E+02,-1.54E+01,7.80E+00,7.80E+00,-1.54E+01,1.31E+02,5.63E+00,-4.33E+00,-4.33E+00,5.63E+00,1.11E+02,-3.12E+01,1.66E+01,1.66E+01,-3.12E+01,1.43E+02,-8.45E+00,3.04E+00,3.04E+00,-8.45E+00,1.52E+02,2.07E+01,-3.66E+01,-3.66E+01,2.07E+01,1.51E+02,-5.99E+00,-1.91E+01,-1.91E+01,-5.99E+00,1.63E+02,-1.10E+01,-1.07E+01,-1.07E+01,-1.10E+01,1.17E+02,-3.93E+00,1.79E+01,1.79E+01,-3.93E+00,1.12E+02,-1.39E+01,2.55E+01,2.55E+01,-1.39E+01,2.25E+02,-9.68E-01,-2.77E+01,-2.77E+01,-9.68E-01,9.74E+01,-1.90E+01,4.07E+01,4.07E+01,-1.90E+01,1.30E+02,-2.33E+01,-2.18E+01,-2.18E+01,-2.33E+01,1.82E+02,2.57E+00,-3.16E+01,-3.16E+01,2.57E+00"  # Replace ... with real values
            happy_sample = "4.62E+00,3.03E+01,-3.56E+02,1.56E+01,2.63E+01,1.07E+00,4.11E-01,-1.57E+01,2.06E+00,3.15E+00,2.15E+00,2.95E+01,-3.53E+02,1.44E+01,2.15E+01,5.98E+00,3.07E+01,-3.43E+02,1.47E+01,2.79E+01,3.17E+00,3.22E+01,-3.68E+02,1.59E+01,3.64E+01,7.08E+00,2.88E+01,-3.59E+02,1.73E+01,1.96E+01,-3.83E+00,-1.23E+00,-1.08E+01,-3.63E-01,-6.41E+00,-1.03E+00,-2.78E+00,1.46E+01,-1.54E+00,-1.49E+01,-4.94E+00,6.64E-01,5.82E+00,-2.92E+00,1.90E+00,2.80E+00,-1.55E+00,2.55E+01,-1.18E+00,-8.51E+00,-1.11E+00,1.89E+00,1.66E+01,-2.55E+00,8.31E+00,-3.91E+00,3.44E+00,-8.82E+00,-1.37E+00,1.68E+01,2.81E+01,7.80E+00,2.03E+02,3.99E+01,3.61E+01,-1.69E+00,-8.76E-01,8.12E+00,-1.72E+00,7.47E-01,1.00E+04,5.74E+01,4.17E+06,4.04E+04,8.75E+03,1.18E+06,1.15E+04,3.69E+09,4.13E+06,5.46E+06,2.70E+07,3.40E+04,7.05E+11,1.84E+08,1.22E+08,2.27E+09,3.32E+06,4.47E+14,1.38E+10,4.07E+10,6.14E+01,5.20E+01,1.31E+02,9.36E+01,1.37E+02,9.59E-01,2.73E+00,-9.42E+01,-9.62E+00,-5.81E+00,5.39E+01,4.93E+01,1.31E+02,9.36E+01,1.37E+02,6.04E+01,4.39E+01,1.65E+01,8.27E+01,1.05E+02,5.73E+01,5.20E+01,2.19E+01,8.34E+01,1.02E+02,6.14E+01,5.00E+01,3.65E+01,8.40E+01,1.31E+02,-6.54E+00,5.34E+00,1.14E+02,1.09E+01,3.19E+01,-3.45E+00,-2.73E+00,1.09E+02,1.03E+01,3.44E+01,-7.50E+00,-7.02E-01,9.42E+01,9.62E+00,5.81E+00,3.10E+00,-8.07E+00,-5.38E+00,-6.72E-01,2.46E+00,-9.59E-01,-6.04E+00,-2.00E+01,-1.31E+00,-2.61E+01,-4.06E+00,2.03E+00,-1.46E+01,-6.35E-01,-2.86E+01,-4.20E+01,9.96E+00,-7.27E+02,-3.80E+01,-7.95E+01,5.51E+00,3.81E+00,-5.02E+01,9.64E+00,3.58E+01,-4.20E+01,9.96E+00,-6.77E+02,-3.80E+01,-4.48E+01,-4.20E+01,1.54E+01,-6.76E+02,-3.33E+01,-7.95E+01,-3.65E+01,1.56E+01,-6.98E+02,-2.80E+01,-3.95E+01,-3.15E+01,1.38E+01,-7.27E+02,-2.84E+01,-4.36E+01,-6.07E-02,-5.40E+00,-9.92E-02,-4.76E+00,3.47E+01,-5.51E+00,-5.69E+00,2.15E+01,-9.98E+00,-5.22E+00,-1.06E+01,-3.81E+00,5.02E+01,-9.64E+00,-1.14E+00,-5.44E+00,-2.84E-01,2.16E+01,-5.22E+00,-3.99E+01,-1.05E+01,1.59E+00,5.03E+01,-4.87E+00,-3.58E+01,-5.07E+00,1.87E+00,2.87E+01,3.43E-01,4.08E+00,2.04E+04,-4.31E+03,3.41E+01,2.12E+03,1.21E+03,7.86E+01,9.61E+02,4.61E+02,-6.14E+03,-4.67E+03,9.97E+02,4.17E+02,-4.31E+03,2.14E+04,6.45E+02,1.55E+03,-3.60E+03,1.36E+03,2.76E+02,-5.78E+02,3.42E+04,-1.76E+03,5.39E+02,2.36E+02,3.41E+01,6.45E+02,5.10E+01,6.29E+01,-1.73E+02,1.45E+02,-3.02E+01,5.60E+00,9.74E+02,-9.31E+02,-1.06E+01,2.04E+01,2.12E+03,1.55E+03,6.29E+01,1.96E+03,1.81E+02,-3.38E+02,-3.42E+02,-1.91E+02,-2.47E+03,3.09E+03,1.73E+02,6.51E+01,1.21E+03,-3.60E+03,-1.73E+02,1.81E+02,1.22E+03,-8.69E+02,2.23E+02,-1.48E+02,-7.83E+03,3.12E+03,-8.45E+01,1.36E+02,7.86E+01,1.36E+03,1.45E+02,-3.38E+02,-8.69E+02,2.54E+03,-4.73E+02,5.05E+02,4.83E+03,-5.34E+03,4.60E+02,-5.29E+02,9.61E+02,2.76E+02,-3.02E+01,-3.42E+02,2.23E+02,-4.73E+02,8.07E+02,-7.92E+01,1.26E+03,4.90E+02,-3.60E+01,2.84E+02,4.61E+02,-5.78E+02,5.60E+00,-1.91E+02,-1.48E+02,5.05E+02,-7.92E+01,2.35E+02,7.53E+02,-7.92E+02,1.33E+02,-1.63E+02,-6.14E+03,3.42E+04,9.74E+02,-2.47E+03,-7.83E+03,4.83E+03,1.26E+03,7.53E+02,8.80E+04,-8.98E+03,1.16E+03,7.58E+02,-4.67E+03,-1.76E+03,-9.31E+02,3.09E+03,3.12E+03,-5.34E+03,4.90E+02,-7.92E+02,-8.98E+03,4.33E+04,5.92E+02,-1.98E+02,9.97E+02,5.39E+02,-1.06E+01,1.73E+02,-8.45E+01,4.60E+02,-3.60E+01,1.33E+02,1.16E+03,5.92E+02,3.38E+02,-2.74E+02,4.17E+02,2.36E+02,2.04E+01,6.51E+01,1.36E+02,-5.29E+02,2.84E+02,-1.63E+02,7.58E+02,-1.98E+02,-2.74E+02,5.85E+02,1.05E+05,4.42E+04,1.92E+04,7.88E+03,2.49E+03,8.45E+02,4.73E+02,1.96E+02,1.06E+02,2.99E+01,-2.08E-12,1.20E+01,9.73E+00,8.37E+00,-1.13E+01,5.74E+00,4.79E+00,7.10E+00,5.57E+00,-6.04E+00,1.10E+01,1.06E+01,-3.60E-01,5.53E+00,-4.39E-01,2.85E+00,-1.86E+00,4.07E-01,-1.37E-02,-3.38E-01,2.20E+00,7.58E-01,-9.36E-02,2.74E-01,-1.00E+00,6.98E-01,1.44E+00,2.67E+00,1.63E-01,9.73E-01,6.58E-01,-7.12E-02,-6.66E-02,-3.25E-01,-9.42E-02,7.09E-01,-9.49E-01,-3.78E-02,-1.29E+00,-3.05E+00,2.21E-01,1.09E-01,6.75E+00,1.72E-01,2.64E-02,-6.24E-02,-2.28E+00,2.22E+00,-2.43E-01,-4.44E-01,-1.45E+00,-4.27E-01,-7.19E-02,7.33E-01,1.14E+01,-5.50E-01,3.02E-01,7.25E-01,4.34E-01,4.43E-01,-2.96E+00,-5.88E-01,4.09E-01,1.76E+00,-4.17E-01,-5.41E-01,1.19E+00,-1.27E-01,-1.54E+00,3.94E-01,-1.04E-02,-5.87E-02,-7.87E+00,4.13E-01,-2.43E-01,1.80E+00,-6.42E-02,8.33E-01,-1.21E-01,6.69E-02,0,4.98E+00,0,0,0,2.49E+05,2.06E+04,1.23E+05,2.62E+05,3.27E+03,1.29E+05,2.74E+05,-4.59E+03,2.19E+05,1.89E+05,3.53E+04,2.55E+05,1.61E+05,8.61E+03,1.16E+05,3.32E+05,2.94E+04,2.51E+05,3.19E+05,1.13E+04,1.44E+05,2.71E+05,5.97E+04,2.67E+05,2.88E+05,6.41E+04,2.56E+05,2.36E+05,6.18E+03,2.15E+05,3.16E+05,1.72E+04,3.10E+05,2.49E+05,1.16E+04,1.72E+05,3.04E+05,8.71E+03,2.33E+05,3.12E+05,1.75E+04,2.00E+05,2.12E+05,1.54E+04,1.55E+05,2.54E+05,2.61E+04,1.87E+05,2.10E+05,1.03E+04,1.71E+05,2.91E+05,1.38E+04,2.85E+05,1.96E+05,5.63E+03,9.39E+04,3.68E+05,1.50E+04,1.91E+05,3.22E+05,7.02E+04,3.84E+05,1.95E+05,2.73E+03,1.53E+05,2.23E+05,1.91E+04,1.66E+05,2.25E+05,9.30E+03,1.86E+05,2.76E+05,9.23E+03,2.78E+05,-5.20E+02,3.98E+02,-2.01E+02,-2.01E+02,3.98E+02,-5.22E+01,1.21E+02,-6.40E+01,-6.40E+01,1.21E+02,-4.12E+02,3.30E+02,-2.29E+02,-2.29E+02,3.30E+02,-5.08E+02,3.31E+02,-1.76E+02,-1.76E+02,3.31E+02,8.68E+01,-4.06E+01,7.14E+01,7.14E+01,-4.06E+01,-3.05E+02,2.50E+02,-1.25E+02,-1.25E+02,2.50E+02,-4.48E+02,4.15E+02,-2.49E+02,-2.49E+02,4.15E+02,1.95E+02,-7.51E+01,4.56E+01,4.56E+01,-7.51E+01,-5.22E+02,4.10E+02,-1.84E+02,-1.84E+02,4.10E+02,-4.06E+02,3.79E+02,-1.53E+02,-1.53E+02,3.79E+02,2.98E+01,1.26E+02,-7.97E+00,-7.97E+00,1.26E+02,-3.87E+02,3.65E+02,-2.52E+02,-2.52E+02,3.65E+02,-2.02E+02,3.38E+02,-2.12E+02,-2.12E+02,3.38E+02,4.18E+02,-8.90E+01,-1.97E+00,-1.97E+00,-8.90E+01,-3.75E+02,3.23E+02,-2.10E+02,-2.10E+02,3.23E+02,-5.07E+02,5.39E+02,-2.90E+02,-2.90E+02,5.39E+02,4.03E+01,1.81E+02,-7.45E+01,-7.45E+01,1.81E+02,-5.12E+02,4.28E+02,-2.26E+02,-2.26E+02,4.28E+02,-4.14E+02,3.79E+02,-2.16E+02,-2.16E+02,3.79E+02,1.41E+02,6.04E+01,-9.31E+00,-9.31E+00,6.04E+01,-3.35E+02,3.10E+02,-1.59E+02,-1.59E+02,3.10E+02,-5.23E+02,4.82E+02,-2.16E+02,-2.16E+02,4.82E+02,-1.10E+02,1.92E+02,-2.56E+00,-2.56E+00,1.92E+02,-5.07E+02,4.29E+02,-1.90E+02,-1.90E+02,4.29E+02,-4.95E+02,4.11E+02,-1.86E+02,-1.86E+02,4.11E+02,-4.10E+01,1.80E+02,-3.60E+01,-3.60E+01,1.80E+02,-4.91E+02,3.85E+02,-1.73E+02,-1.73E+02,3.85E+02,-4.27E+02,4.36E+02,-2.44E+02,-2.44E+02,4.36E+02,7.05E+01,4.80E+01,2.35E+01,2.35E+01,4.80E+01,-5.52E+02,4.48E+02,-2.41E+02,-2.41E+02,4.48E+02,-5.52E+02,4.44E+02,-2.49E+02,-2.49E+02,4.44E+02,-1.26E+01,1.53E+02,-5.08E+01,-5.08E+01,1.53E+02,-6.14E+02,4.67E+02,-1.92E+02,-1.92E+02,4.67E+02,-4.45E+02,3.92E+02,-2.20E+02,-2.20E+02,3.92E+02,1.86E+02,8.28E+00,-9.64E+00,-9.64E+00,8.28E+00,-3.65E+02,2.90E+02,-1.74E+02,-1.74E+02,2.90E+02,-7.04E+02,5.63E+02,-2.36E+02,-2.36E+02,5.63E+02,-1.95E+02,2.35E+02,-5.74E+01,-5.74E+01,2.35E+02,-4.25E+02,3.87E+02,-2.27E+02,-2.27E+02,3.87E+02,-4.45E+02,4.16E+02,-2.13E+02,-2.13E+02,4.16E+02,1.69E+02,5.05E+01,1.46E+00,1.46E+00,5.05E+01,-3.43E+02,2.43E+02,-1.77E+02,-1.77E+02,2.43E+02,-4.29E+02,4.20E+02,-2.23E+02,-2.23E+02,4.20E+02,2.29E+00,1.05E+02,2.25E+01,2.25E+01,1.05E+02,-4.03E+02,3.47E+02,-2.16E+02,-2.16E+02,3.47E+02,-3.20E+02,3.38E+02,-2.19E+02,-2.19E+02,3.38E+02,2.06E+02,9.10E+01,-4.27E+01,-4.27E+01,9.10E+01,-3.00E+02,3.05E+02,-2.24E+02,-2.24E+02,3.05E+02,-5.31E+02,4.84E+02,-2.29E+02,-2.29E+02,4.84E+02,1.94E+02,-3.16E-01,-1.20E+01,-1.20E+01,-3.16E-01,-3.89E+02,3.94E+02,-2.09E+02,-2.09E+02,3.94E+02,-4.75E+02,4.46E+02,-1.99E+02,-1.99E+02,4.46E+02,5.95E+01,9.71E+01,-4.28E+01,-4.28E+01,9.71E+01,-4.32E+02,4.21E+02,-2.20E+02,-2.20E+02,4.21E+02,-3.25E+02,3.68E+02,-1.97E+02,-1.97E+02,3.68E+02,1.02E+02,1.87E+01,1.52E+01,1.52E+01,1.87E+01,-2.24E+02,2.45E+02,-1.82E+02,-1.82E+02,2.45E+02,-6.74E+02,5.68E+02,-2.51E+02,-2.51E+02,5.68E+02,2.11E+02,6.85E+01,-2.66E+01,-2.66E+01,6.85E+01,-3.81E+02,3.34E+02,-1.64E+02,-1.64E+02,3.34E+02,-4.77E+02,3.84E+02,-1.69E+02,-1.69E+02,3.84E+02,-1.82E+02,1.61E+02,2.04E+01,2.04E+01,1.61E+02,-5.96E+02,4.02E+02,-2.03E+02,-2.03E+02,4.02E+02,-3.59E+02,3.50E+02,-2.02E+02,-2.02E+02,3.50E+02,-9.95E+01,1.69E+02,-5.62E+01,-5.62E+01,1.69E+02,-5.40E+02,4.62E+02,-2.36E+02,-2.36E+02,4.62E+02,-4.37E+02,3.98E+02,-1.96E+02,-1.96E+02,3.98E+02,8.01E+01,3.28E+01,-1.95E+01,-1.95E+01,3.28E+01,-3.15E+02,2.95E+02,-1.69E+02,-1.69E+02,2.95E+02,-4.28E+02,4.27E+02,-2.18E+02,-2.18E+02,4.27E+02,2.24E+02,-5.95E+00,1.12E+01,1.12E+01,-5.95E+00,-4.77E+02,4.47E+02,-2.12E+02,-2.12E+02,4.47E+02,-5.08E+02,4.77E+02,-2.16E+02,-2.16E+02,4.77E+02,-6.23E+01,1.70E+02,-3.46E+01,-3.46E+01,1.70E+02,-4.79E+02,4.42E+02,-2.29E+02,-2.29E+02,4.42E+02,-5.09E+02,3.99E+02,-1.99E+02,-1.99E+02,3.99E+02,1.34E+02,8.65E+01,-1.00E+01,-1.00E+01,8.65E+01,-1.51E+02,2.69E+02,-2.15E+02,-2.15E+02,2.69E+02,-5.52E+02,5.23E+02,-2.82E+02,-2.82E+02,5.23E+02,-1.08E+02,1.90E+02,-4.17E+01,-4.17E+01,1.90E+02,-3.76E+02,3.56E+02,-2.00E+02,-2.00E+02,3.56E+02,-5.63E+02,4.45E+02,-2.20E+02,-2.20E+02,4.45E+02,4.94E+01,1.17E+02,-8.26E+01,-8.26E+01,1.17E+02,-3.97E+02,3.51E+02,-2.44E+02,-2.44E+02,3.51E+02,-3.25E+02,3.82E+02,-2.22E+02,-2.22E+02,3.82E+02,-8.69E+01,1.81E+02,-8.33E+01,-8.33E+01,1.81E+02,-5.27E+02,4.84E+02,-2.91E+02,-2.91E+02,4.84E+02,-3.48E+02,3.66E+02,-2.13E+02,-2.13E+02,3.66E+02,1.42E+02,4.91E+01,-1.85E+00,-1.85E+00,4.91E+01,-1.53E+02,2.60E+02,-2.11E+02,-2.11E+02,2.60E+02,-4.21E+02,4.68E+02,-2.43E+02,-2.43E+02,4.68E+02,3.48E+01,5.95E+01,1.82E+01,1.82E+01,5.95E+01,-4.63E+02,3.88E+02,-2.14E+02,-2.14E+02,3.88E+02,-7.45E+02,5.61E+02,-2.39E+02,-2.39E+02,5.61E+02,1.56E+02,3.28E+01,-1.05E+01,-1.05E+01,3.28E+01,-3.47E+02,3.45E+02,-2.01E+02,-2.01E+02,3.45E+02,-4.48E+02,4.00E+02,-2.03E+02,-2.03E+02,4.00E+02,-1.10E+02,1.57E+02,-4.38E+01,-4.38E+01,1.57E+02,-4.80E+02,4.35E+02,-2.55E+02,-2.55E+02,4.35E+02,-5.09E+02,5.13E+02,-2.48E+02,-2.48E+02,5.13E+02,-5.77E+01,2.09E+02,-8.68E+01,-8.68E+01,2.09E+02,-5.36E+02,4.12E+02,-2.14E+02,-2.14E+02,4.12E+02,-3.82E+02,4.09E+02,-1.67E+02,-1.67E+02,4.09E+02,2.45E+02,-3.23E+00,-1.04E+01,-1.04E+01,-3.23E+00,-3.43E+02,3.15E+02,-1.67E+02,-1.67E+02,3.15E+02,-5.73E+02,4.48E+02,-2.01E+02,-2.01E+02,4.48E+02,1.19E+02,3.38E+01,-1.71E+01,-1.71E+01,3.38E+01,-4.88E+02,4.67E+02,-2.32E+02,-2.32E+02,4.67E+02,-4.95E+02,4.39E+02,-1.91E+02,-1.91E+02,4.39E+02,6.18E+01,1.33E+02,-4.94E+01,-4.94E+01,1.33E+02,-4.36E+02,3.77E+02,-2.34E+02,-2.34E+02,3.77E+02,-3.81E+02,3.82E+02,-2.39E+02,-2.39E+02,3.82E+02,1.58E+02,1.37E+01,1.75E+01,1.75E+01,1.37E+01,-4.92E+02,3.99E+02,-2.05E+02,-2.05E+02,3.99E+02,-6.61E+02,5.40E+02,-2.40E+02,-2.40E+02,5.40E+02,-1.53E+02,1.90E+02,-3.58E+00,-3.58E+00,1.90E+02,-7.05E+02,4.98E+02,-2.24E+02,-2.24E+02,4.98E+02,-3.93E+02,3.78E+02,-2.03E+02,-2.03E+02,3.78E+02,1.39E+02,4.21E+01,1.97E+01,1.97E+01,4.21E+01,-3.64E+02,3.18E+02,-1.39E+02,-1.39E+02,3.18E+02,-6.99E+02,5.58E+02,-2.23E+02,-2.23E+02,5.58E+02,-1.57E+02,2.52E+02,-1.94E+01,-1.94E+01,2.52E+02,-4.70E+02,4.25E+02,-1.98E+02,-1.98E+02,4.25E+02,-2.74E+02,3.24E+02,-2.23E+02,-2.23E+02,3.24E+02,1.66E+02,3.30E+01,-1.63E+01,-1.63E+01,3.30E+01,-3.73E+02,3.04E+02,-1.56E+02,-1.56E+02,3.04E+02,-5.66E+02,4.55E+02,-1.94E+02,-1.94E+02,4.55E+02,6.32E+01,5.69E+01,1.19E+01,1.19E+01,5.69E+01,-6.11E+02,4.71E+02,-2.13E+02,-2.13E+02,4.71E+02,-4.24E+02,4.24E+02,-2.23E+02,-2.23E+02,4.24E+02,3.12E+02,3.65E+00,-7.36E+01,-7.36E+01,3.65E+00,-2.56E+02,2.60E+02,-2.02E+02,-2.02E+02,2.60E+02,-5.15E+02,4.50E+02,-2.15E+02,-2.15E+02,4.50E+02,3.18E+01,7.35E+01,2.37E+01,2.37E+01,7.35E+01,-4.66E+02,3.89E+02,-2.27E+02,-2.27E+02,3.89E+02,-6.67E+02,5.60E+02,-2.65E+02,-2.65E+02,5.60E+02,-9.67E+01,2.44E+02,-9.58E+01,-9.58E+01,2.44E+02,-7.17E+02,5.99E+02,-2.55E+02,-2.55E+02,5.99E+02,-4.62E+02,3.96E+02,-1.95E+02,-1.95E+02,3.96E+02,1.59E+02,-8.29E+00,2.55E+01,2.55E+01,-8.29E+00,-1.94E+02,2.38E+02,-1.88E+02,-1.88E+02,2.38E+02,-4.13E+02,4.02E+02,-2.14E+02,-2.14E+02,4.02E+02,2.82E+01,1.51E+02,-7.52E+01,-7.52E+01,1.51E+02,-5.07E+02,3.79E+02,-1.99E+02,-1.99E+02,3.79E+02,-3.91E+02,3.74E+02,-1.48E+02,-1.48E+02,3.74E+02,1.37E+02,-2.35E+00,4.66E+01,4.66E+01,-2.35E+00,-2.82E+02,2.75E+02,-1.13E+02,-1.13E+02,2.75E+02,-4.75E+02,4.15E+02,-1.97E+02,-1.97E+02,4.15E+02,1.79E+02,2.64E+01,-1.29E+01,-1.29E+01,2.64E+01,-4.74E+02,4.42E+02,-2.22E+02,-2.22E+02,4.42E+02,3.84E+00,3.04E+01,-3.51E+02,1.56E+01,2.61E+01,-2.73E+00,-2.22E-01,2.54E+01,-1.90E+00,-3.84E+00,3.31E+00,3.24E+01,-3.68E+02,1.58E+01,3.68E+01,7.04E+00,2.88E+01,-3.59E+02,1.73E+01,1.95E+01,1.04E+00,2.98E+01,-3.44E+02,1.52E+01,1.90E+01,3.88E+00,3.08E+01,-3.33E+02,1.42E+01,2.92E+01,-3.73E+00,3.59E+00,-9.24E+00,-1.45E+00,1.73E+01,2.28E+00,2.59E+00,-2.44E+01,6.32E-01,1.77E+01,-5.67E-01,1.51E+00,-3.56E+01,1.68E+00,7.58E+00,6.01E+00,-1.00E+00,-1.52E+01,2.08E+00,4.59E-01,3.16E+00,-2.08E+00,-2.63E+01,3.13E+00,-9.68E+00,-2.84E+00,-1.08E+00,-1.11E+01,1.05E+00,-1.01E+01,2.81E+01,7.15E+00,2.10E+02,3.96E+01,3.46E+01,2.01E+00,-4.87E-01,3.83E+00,1.13E+00,-3.34E+00,1.02E+04,2.75E+01,4.39E+06,4.00E+04,1.05E+04,1.22E+06,8.51E+03,4.04E+09,4.05E+06,3.61E+06,3.02E+07,2.96E+04,7.10E+11,1.82E+08,1.09E+08,2.56E+09,2.27E+06,4.68E+14,1.34E+10,1.84E+10,6.86E+01,5.18E+01,8.62E+01,8.61E+01,1.29E+02,7.26E+00,-2.31E+00,4.87E+01,1.26E+00,-3.59E+01,5.44E+01,5.18E+01,1.90E+01,8.33E+01,1.09E+02,6.13E+01,4.99E+01,3.75E+01,8.48E+01,1.29E+02,5.36E+01,3.74E+01,6.84E+01,8.61E+01,9.30E+01,6.86E+01,4.95E+01,8.62E+01,8.53E+01,8.65E+01,-6.95E+00,1.84E+00,-1.85E+01,-1.50E+00,-1.96E+01,7.77E-01,1.44E+01,-4.94E+01,-2.76E+00,1.63E+01,-1.42E+01,2.31E+00,-6.72E+01,-1.97E+00,2.27E+01,7.73E+00,1.25E+01,-3.09E+01,-1.26E+00,3.59E+01,-7.26E+00,4.72E-01,-4.87E+01,-4.70E-01,4.24E+01,-1.50E+01,-1.20E+01,-1.78E+01,7.92E-01,6.45E+00,-4.58E+01,1.37E+01,-7.27E+02,-3.30E+01,-4.17E+01,-1.03E+01,-1.19E-01,2.69E+01,-4.38E+00,1.50E+00,-3.56E+01,1.68E+01,-7.02E+02,-2.79E+01,-3.69E+01,-3.15E+01,1.38E+01,-7.27E+02,-2.87E+01,-4.17E+01,-4.58E+01,1.89E+01,-7.00E+02,-3.30E+01,-4.02E+01,-4.16E+01,1.37E+01,-6.58E+02,-3.03E+01,-2.67E+01,-4.05E+00,2.94E+00,2.49E+01,8.18E-01,4.84E+00,1.03E+01,-2.17E+00,-1.94E+00,5.19E+00,3.34E+00,6.02E+00,3.06E+00,-4.40E+01,2.40E+00,-1.02E+01,1.43E+01,-5.11E+00,-2.69E+01,4.38E+00,-1.50E+00,1.01E+01,1.19E-01,-6.89E+01,1.58E+00,-1.50E+01,-4.25E+00,5.23E+00,-4.21E+01,-2.79E+00,-1.35E+01,2.23E+04,-4.06E+03,-1.96E+02,8.30E+02,1.90E+03,-1.72E+03,8.54E+01,2.08E+03,-7.94E+03,-4.72E+03,-6.37E+02,-1.09E+03,-4.06E+03,1.86E+04,-1.08E+03,7.84E+02,-2.98E+03,-3.56E+02,-1.03E+02,3.78E+02,3.41E+04,-1.54E+03,-2.78E+02,-9.56E+02,-1.96E+02,-1.08E+03,2.35E+02,-1.12E+02,2.43E+02,-2.11E+01,-4.35E+01,-6.39E+01,-2.63E+03,1.26E+03,1.33E+01,-2.83E+01,8.30E+02,7.84E+02,-1.12E+02,1.04E+03,7.22E+01,-8.04E+02,1.18E+02,2.21E+02,1.76E+03,1.23E+03,-1.09E+01,-9.53E+01,1.90E+03,-2.98E+03,2.43E+02,7.22E+01,7.97E+02,-2.97E+02,-2.10E+02,6.47E+01,-6.77E+03,1.85E+03,8.91E+01,-8.04E+01,-1.72E+03,-3.56E+02,-2.11E+01,-8.04E+02,-2.97E+02,2.16E+03,-1.26E+02,-4.18E+02,-6.16E+02,-4.36E+03,3.42E+02,4.20E+02,8.54E+01,-1.03E+02,-4.35E+01,1.18E+02,-2.10E+02,-1.26E+02,7.22E+02,-1.98E+02,-3.69E+02,-8.23E+01,-2.83E+02,4.31E+02,2.08E+03,3.78E+02,-6.39E+01,2.21E+02,6.47E+01,-4.18E+02,-1.98E+02,5.53E+02,2.75E+03,-4.80E+02,-4.86E+00,-3.95E+02,-7.94E+03,3.41E+04,-2.63E+03,1.76E+03,-6.77E+03,-6.16E+02,-3.69E+02,2.75E+03,9.68E+04,-7.40E+03,-4.92E+02,-2.57E+03,-4.72E+03,-1.54E+03,1.26E+03,1.23E+03,1.85E+03,-4.36E+03,-8.23E+01,-4.80E+02,-7.40E+03,3.57E+04,-5.49E+02,-5.33E+02,-6.37E+02,-2.78E+02,1.33E+01,-1.09E+01,8.91E+01,3.42E+02,-2.83E+02,-4.86E+00,-4.92E+02,-5.49E+02,2.57E+02,-5.30E+01,-1.09E+03,-9.56E+02,-2.83E+01,-9.53E+01,-8.04E+01,4.20E+02,4.31E+02,-3.95E+02,-2.57E+03,-5.33E+02,-5.30E+01,5.24E+02,1.12E+05,3.75E+04,2.08E+04,5.75E+03,1.63E+03,1.23E+03,6.57E+02,1.47E+02,8.88E+01,-9.87E-13,1.07E+01,4.31E+01,9.74E+00,8.94E+00,-2.15E+00,6.51E+00,-2.35E-01,6.13E+00,1.80E-01,4.64E+00,1.12E+01,1.04E+01,3.41E+00,-9.03E+00,-9.23E-02,-1.42E-01,1.12E-01,2.14E-01,-1.76E+00,-2.36E+00,-6.56E-02,2.93E-01,-1.35E-01,-1.29E-01,-3.39E+00,-9.94E-01,1.26E-01,4.94E+00,-6.73E-01,-5.10E+00,-2.84E-01,1.22E-01,-3.18E-02,-8.71E-02,-1.20E-01,1.59E-01,-5.57E-01,2.58E+00,1.57E-02,-3.97E-02,2.80E-02,3.91E-02,2.97E-01,-3.78E-01,1.10E+00,-4.23E-02,6.25E+00,1.62E-01,-2.46E-01,-4.30E-01,1.31E+00,-1.14E+00,8.79E-02,-6.41E-02,-1.61E-01,9.75E-02,3.62E-01,1.39E+00,9.78E+00,9.02E-01,-3.09E-01,-2.91E-01,1.04E-01,2.39E+00,3.86E+00,4.60E-01,9.54E-01,1.36E-01,5.12E-01,6.17E+00,-1.77E-01,3.11E-02,-2.46E+00,6.49E-01,-2.48E-01,-1.21E-02,-1.02E+01,-5.74E-01,-2.20E-01,-1.48E+00,0,4.98E+00,0,0,0,2.78E+05,1.35E+04,1.12E+05,2.62E+05,3.98E+03,1.47E+05,3.43E+05,4.78E+04,3.16E+05,1.87E+05,-7.86E+03,1.90E+05,2.44E+05,1.51E+04,1.11E+05,2.44E+05,1.13E+04,2.31E+05,3.39E+05,8.14E+03,2.13E+05,2.09E+05,6.03E+04,2.89E+05,2.53E+05,9.88E+03,1.86E+05,2.78E+05,5.08E+03,1.94E+05,2.41E+05,7.82E+03,1.70E+05,3.26E+05,2.96E+04,1.84E+05,1.86E+05,9.06E+03,2.40E+05,2.81E+05,-4.58E+03,3.27E+05,2.52E+05,1.61E+04,1.81E+05,3.18E+05,-3.92E+03,1.94E+05,2.42E+05,1.17E+04,1.47E+05,3.19E+05,1.35E+04,2.41E+05,2.01E+05,5.85E+03,1.43E+05,2.41E+05,-2.84E+03,2.05E+05,3.52E+05,8.92E+03,4.00E+05,2.30E+05,8.77E+03,1.11E+05,2.14E+05,1.54E+04,2.21E+05,2.66E+05,1.27E+04,2.21E+05,2.12E+05,6.38E+03,1.57E+05,-5.08E+02,4.38E+02,-2.22E+02,-2.22E+02,4.38E+02,1.43E+02,5.94E+01,5.05E+00,5.05E+00,5.94E+01,-1.59E+02,2.90E+02,-2.27E+02,-2.27E+02,2.90E+02,-5.46E+02,5.06E+02,-2.73E+02,-2.73E+02,5.06E+02,-1.13E+02,2.04E+02,-5.01E+01,-5.01E+01,2.04E+02,-3.72E+02,3.43E+02,-1.92E+02,-1.92E+02,3.43E+02,-5.67E+02,4.56E+02,-2.26E+02,-2.26E+02,4.56E+02,5.24E+01,1.07E+02,-7.64E+01,-7.64E+01,1.07E+02,-4.00E+02,3.61E+02,-2.50E+02,-2.50E+02,3.61E+02,-3.23E+02,3.73E+02,-2.17E+02,-2.17E+02,3.73E+02,-8.88E+01,1.89E+02,-8.82E+01,-8.82E+01,1.89E+02,-5.25E+02,4.76E+02,-2.86E+02,-2.86E+02,4.76E+02,-3.49E+02,3.73E+02,-2.17E+02,-2.17E+02,3.73E+02,1.43E+02,4.25E+01,2.34E+00,2.34E+00,4.25E+01,-1.54E+02,2.67E+02,-2.15E+02,-2.15E+02,2.67E+02,-4.20E+02,4.62E+02,-2.39E+02,-2.39E+02,4.62E+02,3.41E+01,6.50E+01,1.46E+01,1.46E+01,6.50E+01,-4.63E+02,3.82E+02,-2.10E+02,-2.10E+02,3.82E+02,-7.46E+02,5.66E+02,-2.42E+02,-2.42E+02,5.66E+02,1.56E+02,2.80E+01,-7.27E+00,-7.27E+00,2.80E+01,-3.47E+02,3.50E+02,-2.04E+02,-2.04E+02,3.50E+02,-4.48E+02,3.96E+02,-1.99E+02,-1.99E+02,3.96E+02,-1.11E+02,1.62E+02,-4.68E+01,-4.68E+01,1.62E+02,-4.80E+02,4.31E+02,-2.52E+02,-2.52E+02,4.31E+02,-5.09E+02,5.17E+02,-2.50E+02,-2.50E+02,5.17E+02,-5.76E+01,2.05E+02,-8.41E+01,-8.41E+01,2.05E+02,-5.37E+02,4.16E+02,-2.17E+02,-2.17E+02,4.16E+02,-3.82E+02,4.05E+02,-1.64E+02,-1.64E+02,4.05E+02,2.45E+02,6.41E-02,-1.29E+01,-1.29E+01,6.41E-02,-3.43E+02,3.12E+02,-1.65E+02,-1.65E+02,3.12E+02,-5.73E+02,4.51E+02,-2.04E+02,-2.04E+02,4.51E+02,1.19E+02,3.08E+01,-1.47E+01,-1.47E+01,3.08E+01,-4.87E+02,4.69E+02,-2.34E+02,-2.34E+02,4.69E+02,-4.95E+02,4.37E+02,-1.88E+02,-1.88E+02,4.37E+02,6.20E+01,1.35E+02,-5.16E+01,-5.16E+01,1.35E+02,-4.36E+02,3.75E+02,-2.32E+02,-2.32E+02,3.75E+02,-3.80E+02,3.84E+02,-2.41E+02,-2.41E+02,3.84E+02,1.57E+02,1.14E+01,1.96E+01,1.96E+01,1.14E+01,-4.92E+02,4.01E+02,-2.07E+02,-2.07E+02,4.01E+02,-6.61E+02,5.38E+02,-2.38E+02,-2.38E+02,5.38E+02,-1.52E+02,1.92E+02,-5.59E+00,-5.59E+00,1.92E+02,-7.06E+02,4.96E+02,-2.22E+02,-2.22E+02,4.96E+02,-3.92E+02,3.80E+02,-2.05E+02,-2.05E+02,3.80E+02,1.39E+02,4.03E+01,2.16E+01,2.16E+01,4.03E+01,-3.63E+02,3.20E+02,-1.41E+02,-1.41E+02,3.20E+02,-7.00E+02,5.57E+02,-2.21E+02,-2.21E+02,5.57E+02,-1.57E+02,2.53E+02,-2.13E+01,-2.13E+01,2.53E+02,-4.71E+02,4.24E+02,-1.96E+02,-1.96E+02,4.24E+02,-2.74E+02,3.25E+02,-2.25E+02,-2.25E+02,3.25E+02,1.65E+02,3.18E+01,-1.45E+01,-1.45E+01,3.18E+01,-3.72E+02,3.05E+02,-1.57E+02,-1.57E+02,3.05E+02,-5.66E+02,4.54E+02,-1.92E+02,-1.92E+02,4.54E+02,6.38E+01,5.79E+01,1.01E+01,1.01E+01,5.79E+01,-6.12E+02,4.70E+02,-2.11E+02,-2.11E+02,4.70E+02,-4.23E+02,4.25E+02,-2.24E+02,-2.24E+02,4.25E+02,3.11E+02,2.98E+00,-7.19E+01,-7.19E+01,2.98E+00,-2.55E+02,2.60E+02,-2.03E+02,-2.03E+02,2.60E+02,-5.16E+02,4.50E+02,-2.14E+02,-2.14E+02,4.50E+02,3.27E+01,7.39E+01,2.20E+01,2.20E+01,7.39E+01,-4.67E+02,3.89E+02,-2.25E+02,-2.25E+02,3.89E+02,-6.66E+02,5.60E+02,-2.66E+02,-2.66E+02,5.60E+02,-9.78E+01,2.44E+02,-9.41E+01,-9.41E+01,2.44E+02,-7.16E+02,5.99E+02,-2.56E+02,-2.56E+02,5.99E+02,-4.63E+02,3.96E+02,-1.94E+02,-1.94E+02,3.96E+02,1.60E+02,-8.63E+00,2.38E+01,2.38E+01,-8.63E+00,-1.95E+02,2.39E+02,-1.87E+02,-1.87E+02,2.39E+02,-4.11E+02,4.01E+02,-2.16E+02,-2.16E+02,4.01E+02,2.67E+01,1.52E+02,-7.35E+01,-7.35E+01,1.52E+02,-5.06E+02,3.78E+02,-2.01E+02,-2.01E+02,3.78E+02,-3.93E+02,3.76E+02,-1.47E+02,-1.47E+02,3.76E+02,1.39E+02,-3.97E+00,4.51E+01,4.51E+01,-3.97E+00,-2.83E+02,2.77E+02,-1.11E+02,-1.11E+02,2.77E+02,-4.74E+02,4.12E+02,-1.98E+02,-1.98E+02,4.12E+02,1.79E+02,2.97E+01,-1.22E+01,-1.22E+01,2.97E+01,-4.78E+02,4.38E+02,-2.22E+02,-2.22E+02,4.38E+02,-4.78E+02,4.34E+02,-2.17E+02,-2.17E+02,4.34E+02,2.29E+02,2.95E+01,-1.49E+01,-1.49E+01,2.95E+01,-2.45E+02,2.98E+02,-1.89E+02,-1.89E+02,2.98E+02,-3.32E+02,3.78E+02,-1.94E+02,-1.94E+02,3.78E+02,2.04E+02,-1.97E+01,1.93E+01,1.93E+01,-1.97E+01,-2.57E+02,3.23E+02,-2.25E+02,-2.25E+02,3.23E+02,-5.30E+02,5.08E+02,-2.49E+02,-2.49E+02,5.08E+02,-2.25E+02,2.12E+02,-3.00E+01,-3.00E+01,2.12E+02,-7.09E+02,5.71E+02,-2.42E+02,-2.42E+02,5.71E+02,-4.08E+02,3.53E+02,-1.84E+02,-1.84E+02,3.53E+02,2.81E+02,-4.25E+01,1.51E+01,1.51E+01,-4.25E+01,-3.35E+02,2.59E+02,-1.51E+02,-1.51E+02,2.59E+02,-5.61E+02,4.57E+02,-1.96E+02,-1.96E+02,4.57E+02,4.85E+01,1.16E+02,-6.56E+00,-6.56E+00,1.16E+02,-3.78E+02,2.97E+02,-1.94E+02,-1.94E+02,2.97E+02,-4.56E+02,3.80E+02,-1.91E+02,-1.91E+02,3.80E+02,1.22E+02,3.06E+01,-2.02E+01,-2.02E+01,3.06E+01,-4.81E+02,3.83E+02,-2.13E+02,-2.13E+02,3.83E+02,-4.74E+02,4.10E+02,-2.03E+02,-2.03E+02,4.10E+02,-1.41E+01,6.47E+01,1.13E+01,1.13E+01,6.47E+01,-5.59E+02,4.23E+02,-2.58E+02,-2.58E+02,4.23E+02,-4.23E+02,3.54E+02,-1.79E+02,-1.79E+02,3.54E+02,-1.79E+02,1.67E+02,-4.45E+01,-4.45E+01,1.67E+02,-5.30E+02,4.64E+02,-2.66E+02,-2.66E+02,4.64E+02,-4.26E+02,3.55E+02,-1.90E+02,-1.90E+02,3.55E+02,1.17E+02,1.08E+01,1.96E+01,1.96E+01,1.08E+01,-3.16E+02,2.84E+02,-2.07E+02,-2.07E+02,2.84E+02,-5.90E+02,4.82E+02,-2.27E+02,-2.27E+02,4.82E+02,-6.09E+00,1.18E+02,7.83E+00,7.83E+00,1.18E+02,-5.15E+02,4.09E+02,-2.45E+02,-2.45E+02,4.09E+02,-4.04E+02,3.60E+02,-1.65E+02,-1.65E+02,3.60E+02,2.21E+02,-7.34E-01,-4.94E+00,-4.94E+00,-7.34E-01,-3.35E+02,2.57E+02,-1.26E+02,-1.26E+02,2.57E+02,-6.46E+02,5.04E+02,-2.20E+02,-2.20E+02,5.04E+02,-4.03E+01,1.05E+02,-2.03E+01,-2.03E+01,1.05E+02,-3.66E+02,3.38E+02,-2.06E+02,-2.06E+02,3.38E+02,-3.25E+02,3.68E+02,-1.76E+02,-1.76E+02,3.68E+02,-3.33E+00,1.21E+02,-8.65E+00,-8.65E+00,1.21E+02,-5.03E+02,3.76E+02,-2.22E+02,-2.22E+02,3.76E+02,-3.53E+02,3.67E+02,-2.50E+02,-2.50E+02,3.67E+02,1.87E+02,-8.52E+00,-4.39E+01,-4.39E+01,-8.52E+00,-4.89E+02,4.42E+02,-2.45E+02,-2.45E+02,4.42E+02,-5.54E+02,5.00E+02,-2.46E+02,-2.46E+02,5.00E+02,6.30E+01,1.28E+02,-5.83E+01,-5.83E+01,1.28E+02,-5.29E+02,3.84E+02,-2.23E+02,-2.23E+02,3.84E+02,-5.06E+02,3.88E+02,-1.78E+02,-1.78E+02,3.88E+02,2.55E+02,4.81E+00,2.04E+01,2.04E+01,4.81E+00,-4.17E+02,3.01E+02,-1.63E+02,-1.63E+02,3.01E+02,-6.39E+02,5.25E+02,-2.69E+02,-2.69E+02,5.25E+02,2.38E+02,1.86E+00,-2.31E+01,-2.31E+01,1.86E+00,-3.63E+02,3.43E+02,-1.61E+02,-1.61E+02,3.43E+02,-5.85E+02,4.70E+02,-2.00E+02,-2.00E+02,4.70E+02,1.08E+02,5.46E+01,-3.26E+01,-3.26E+01,5.46E+01,-3.29E+02,3.45E+02,-2.33E+02,-2.33E+02,3.45E+02,-3.25E+02,3.44E+02,-1.86E+02,-1.86E+02,3.44E+02,-2.87E+01,1.80E+02,-6.34E+01,-6.34E+01,1.80E+02,-3.80E+02,3.67E+02,-2.46E+02,-2.46E+02,3.67E+02,-4.50E+02,3.73E+02,-1.56E+02,-1.56E+02,3.73E+02,2.75E+02,-6.15E+01,-9.09E+00,-9.09E+00,-6.15E+01,-4.27E+02,3.25E+02,-1.91E+02,-1.91E+02,3.25E+02,-5.42E+02,4.31E+02,-1.89E+02,-1.89E+02,4.31E+02,1.47E+02,-8.00E+00,6.23E+01,6.23E+01,-8.00E+00,-5.22E+02,4.54E+02,-2.16E+02,-2.16E+02,4.54E+02,-4.41E+02,4.03E+02,-1.79E+02,-1.79E+02,4.03E+02,1.80E+02,4.07E+01,4.08E+01,4.08E+01,4.07E+01,-3.52E+02,3.25E+02,-2.41E+02,-2.41E+02,3.25E+02,-3.99E+02,4.10E+02,-1.87E+02,-1.87E+02,4.10E+02,2.47E+02,3.09E+01,-2.44E+01,-2.44E+01,3.09E+01,-4.42E+02,3.88E+02,-1.97E+02,-1.97E+02,3.88E+02,-5.64E+02,5.00E+02,-2.45E+02,-2.45E+02,5.00E+02,-8.88E+01,2.14E+02,-8.88E+01,-8.88E+01,2.14E+02,-6.06E+02,5.09E+02,-2.61E+02,-2.61E+02,5.09E+02,-3.99E+02,3.74E+02,-1.85E+02,-1.85E+02,3.74E+02,7.43E+01,2.35E+01,2.03E+01,2.03E+01,2.35E+01,-2.15E+02,2.80E+02,-1.62E+02,-1.62E+02,2.80E+02"  # Replace ... with real values

            col1, col2, col3 = st.columns(3)
            
            if 'pasted_emotion_data' not in st.session_state:
                st.session_state.pasted_emotion_data = ""
            
            with col1:
                if st.button("😠"):
                    st.session_state.pasted_emotion_data = angry_sample
            
            with col2:
                if st.button("😊"):
                    st.session_state.pasted_emotion_data = happy_sample
            
            with col3:
                if st.button("😐"):
                
                    st.session_state.pasted_emotion_data = neutral_sample

            # Text area for pasting data
            pasted_data = st.text_area(
                "Paste a row of data (comma-separated values) to get emotion predictions.",
                value=st.session_state.pasted_emotion_data,
                height=100
            )

            predict_button = st.button("Predict Emotion")
            prediction_result_placeholder = st.empty()

        # Create a container for visualization
        visualization_container = st.container()

        if pasted_data and predict_button:
            try:
                values = [float(x.strip()) for x in pasted_data.split(',')]
                # Use your actual feature names here if possible!
                feature_names = [f'Feature_{i}' for i in range(len(values))]

                if len(values) == len(feature_names):
                    input_data = dict(zip(feature_names, values))
                    input_df = pd.DataFrame([input_data])
                    
                    # Preprocess the data
                    X = preprocess_emotion_data(input_df)
                    
                    # Make emotion prediction
                    prediction = make_emotion_predictions(emotion_model, emotion_scaler, feature_selector, X)
                    
                    if prediction is not None:
                        predicted_label = prediction[0]
                        result = emotion_labels.get(predicted_label, "Unknown Emotion")

                        # Display prediction result
                        with prediction_result_placeholder.container():
                            st.subheader("Emotion Prediction Result")
                            st.write(f"Predicted Label: {predicted_label}")
                            result_html = f"""
                            <div style="font-size: 1.2em; margin-top: 5px; color: {emotion_colors[predicted_label]};">
                                <strong>Emotion:</strong> {result}
                            </div>
                            """
                            st.markdown(result_html, unsafe_allow_html=True)
                            st.markdown("---")

                    # # Create emotion visualization
                    # with visualization_container:
                    #     st.subheader("Emotion Visualization")
                    #     emotion_counts = {predicted_label: 1}
                    #     for emotion in emotion_labels.keys():
                    #         if emotion != predicted_label:
                    #             emotion_counts[emotion] = 0
                    #     gauge_fig = create_emotion_gauge(emotion_counts)
                    #     st.plotly_chart(gauge_fig, use_container_width=True)

                    #     st.subheader("Signal Features")
                    #     fig = go.Figure()
                    #     fig.add_trace(go.Scatter(
                    #         y=list(input_data.values()),
                    #         mode='lines+markers',
                    #         name='EEG Signal',
                    #         line=dict(color=emotion_colors[predicted_label], width=2),
                    #         marker=dict(size=5, color=emotion_colors[predicted_label])
                    #     ))
                    #     fig.update_layout(
                    #         title='Input EEG Signal',
                    #         xaxis_title='Feature',
                    #         yaxis_title='Value',
                    #         height=300,
                    #         margin=dict(l=10, r=10, t=40, b=80),
                    #         xaxis=dict(
                    #             ticktext=list(input_data.keys()),
                    #             tickvals=list(range(len(input_data))),
                    #             tickangle=45,
                    #             tickfont=dict(size=8)
                    #         )
                    #     )
                    #     st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"Expected {len(feature_names)} values, got {len(values)}")
            except Exception as e:
                st.error(f"Error processing pasted data: {str(e)}")

if __name__ == "__main__":
    main()