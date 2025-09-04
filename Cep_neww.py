import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Page Configuration ---
st.set_page_config(page_title="Citizen Engagement Platform", layout="wide")

# --- Header ---
st.title("Citizen Engagement Platform")

# --- Subheader ---
st.header("Predictive & Automation Features: Issue Escalation and Categorization")

# --- CSV Upload Functionality ---
st.sidebar.title("Upload Data")
st.sidebar.markdown("Please upload your CSV data below.")

# Add a file uploader widget to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Initialize DataFrame variable
df = None

# --- Main Content Area ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")

        # Display the dataset with dataframe keys
        st.subheader("1. Dataset Overview")
        st.write("Here's an overview of the first 10 rows of your uploaded dataset:")
        st.dataframe(df.head(10))
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

        # --- Data Preprocessing for Escalation Prediction ---
        st.subheader("2. Issue Escalation Prediction Model")
        st.write("This section focuses on predicting the likelihood of an issue being escalated.")

        # Convert date columns to datetime objects
        # Using errors='coerce' will turn unparseable dates into NaT
        df['date_reported'] = pd.to_datetime(df['date_reported'], errors='coerce')
        df['date_resolved'] = pd.to_datetime(df['date_resolved'], errors='coerce')

        # Calculate resolution time
        df['resolution_time'] = (df['date_resolved'] - df['date_reported']).dt.days
        # Handle cases where resolution_time might be negative or NaN if dates are invalid or unresolved
        # For unresolved issues, we can impute a large number or handle them as a separate case.
        # Here, we'll impute 0 for simplicity if resolution date is missing or invalid,
        # but in a real scenario, you'd want a more robust approach.
        df['resolution_time'] = df['resolution_time'].apply(lambda x: max(x, 0) if pd.notnull(x) else 0)

        # Feature Engineering for escalation prediction
        # For demonstration, we'll create a simplified 'category' from 'assigned_department'.
        # In a real application, you'd have a proper 'category' column or derive it more meaningfully.
        if 'assigned_department' in df.columns:
            df['category'] = df['assigned_department']
        else:
            st.warning("The 'assigned_department' column is missing, cannot create 'category' for prediction.")
            # If 'assigned_department' is missing, create a placeholder or skip this part
            df['category'] = 'Unknown' # Placeholder

        # Define features (X) and target (y) for escalation prediction
        # Ensure all required columns exist
        required_cols_escalation = ['assigned_department', 'assigned_level', 'resolution_time', 'category']
        if all(col in df.columns for col in required_cols_escalation) and 'escalated' in df.columns:
            features_escalation = required_cols_escalation
            target_escalation = 'escalated'

            # Convert 'escalated' to numerical (1 for Yes, 0 for No)
            df[target_escalation] = df[target_escalation].apply(lambda x: 1 if pd.notnull(x) and str(x).lower() == 'yes' else 0)

            # Handle potential missing values in features for the escalation model
            df_escalation = df[features_escalation + [target_escalation]].dropna()

            if not df_escalation.empty:
                # Identify categorical and numerical features
                categorical_features_escalation = ['assigned_department', 'assigned_level', 'category']
                numerical_features_escalation = ['resolution_time']

                # Check if these columns exist in the processed DataFrame
                categorical_features_escalation = [col for col in categorical_features_escalation if col in df_escalation.columns]
                numerical_features_escalation = [col for col in numerical_features_escalation if col in df_escalation.columns]

                # 9.1. Define features (X) and target (y)
                X = df_escalation[categorical_features_escalation + numerical_features_escalation]
                y = df_escalation[target_escalation]

                # 9.2. Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                # Create preprocessing pipelines for numerical and categorical features
                numerical_transformer_escalation = StandardScaler()
                categorical_transformer_escalation = OneHotEncoder(handle_unknown='ignore')

                # Create a column transformer to apply different transformations to different columns
                preprocessor_escalation = ColumnTransformer(
                    transformers=[
                        ('num', numerical_transformer_escalation, numerical_features_escalation),
                        ('cat', categorical_transformer_escalation, categorical_features_escalation)
                    ],
                    remainder='passthrough' # Keep other columns if any
                )

                # 9.3. & 9.4. Train the logistic regression model
                # Create the full pipeline including preprocessing and the model
                model_pipeline_escalation = Pipeline(steps=[('preprocessor', preprocessor_escalation),
                                                              ('classifier', LogisticRegression(random_state=42, solver='liblinear'))])

                # Fit the model
                model_pipeline_escalation.fit(X_train, y_train)

                # 9.5. Evaluate and display how the model performs
                st.write("### Model Evaluation (Issue Escalation)")
                y_pred_escalation = model_pipeline_escalation.predict(X_test)
                accuracy_escalation = accuracy_score(y_test, y_pred_escalation)
                st.write(f"**Accuracy:** {accuracy_escalation:.2f}")
                st.write("**Classification Report:**")
                # Use try-except for classification_report in case of only one class in predictions
                try:
                    st.text(classification_report(y_test, y_pred_escalation, target_names=['Not Escalated', 'Escalated']))
                except ValueError:
                    st.text("Could not generate classification report (likely due to only one class present in predictions).")


                # Display confusion matrix
                st.write("**Confusion Matrix:**")
                cm_escalation = confusion_matrix(y_test, y_pred_escalation)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm_escalation, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Escalated', 'Escalated'], yticklabels=['Not Escalated', 'Escalated'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(plt.gcf())

                # 9.6. Interpret feature importance (Coefficients)
                st.write("### Feature Importance (Coefficients)")
                try:
                    # Get feature names after one-hot encoding
                    feature_names_out = model_pipeline_escalation.named_steps['preprocessor'].get_feature_names_out()
                    coefficients_escalation = model_pipeline_escalation.named_steps['classifier'].coef_[0]

                    # Create a DataFrame for better readability
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names_out,
                        'Coefficient': coefficients_escalation
                    })

                    # Sort features by absolute coefficient value
                    feature_importance_df['Absolute_Coefficient'] = abs(feature_importance_df['Coefficient'])
                    feature_importance_df = feature_importance_df.sort_values('Absolute_Coefficient', ascending=False)

                    st.dataframe(feature_importance_df[['Feature', 'Coefficient']])
                    st.write("""
                    **Interpretation:**
                    - **Positive Coefficients:** Indicate that an increase in the feature's value (or presence for categorical features) increases the likelihood of escalation.
                    - **Negative Coefficients:** Indicate that an increase in the feature's value decreases the likelihood of escalation.
                    - **Magnitude:** The larger the absolute value of the coefficient, the greater its impact on the prediction.
                    """)
                except Exception as e:
                    st.warning(f"Could not display feature importance: {e}")
            else:
                st.warning("Not enough valid data to train the escalation prediction model after preprocessing.")
        else:
            st.warning("Required columns for escalation prediction are missing in the uploaded dataset.")


        # --- Text Classification for Faster Routing ---
        st.subheader("4. Issue Categorization and Routing")
        st.write("This section automatically categorizes issues based on their descriptions to route them to the correct department.")

        # Check if necessary columns exist for classification
        if 'description' in df.columns and 'assigned_department' in df.columns:
            # 11.1. Show the most frequent assigned department
            st.write("### Department Frequency")
            if not df['assigned_department'].empty:
                department_counts = df['assigned_department'].value_counts()
                st.bar_chart(department_counts)

                # 11.2. Count occurrences of each assigned department and sort them
                st.write("Top 5 Assigned Departments:")
                st.write(department_counts.head(5))
            else:
                st.warning("No 'assigned_department' data available for frequency analysis.")

            # 11.3. Prepare the data for categorization
            # 11.5. Check for missing descriptions or departments
            if df['description'].isnull().any() or df['assigned_department'].isnull().any():
                st.warning("Some issues have missing descriptions or assigned departments. These rows will be excluded from the classification model.")
                df_classification = df.dropna(subset=['description', 'assigned_department']).copy()
            else:
                df_classification = df.copy()

            if not df_classification.empty:
                # 11.4. Use "assigned_department" as our target for classification.
                y_classification = df_classification['assigned_department']

                # 11.7. Get a list of unique assigned departments and create a mapping.
                unique_departments = y_classification.unique()
                department_mapping = {dept: i for i, dept in enumerate(unique_departments)}
                inverse_department_mapping = {i: dept for dept, i in department_mapping.items()}

                # 11.8. Clean the 'description' column for the model.
                # Download stopwords if not already present
                try:
                    stopwords_english = set(stopwords.words('english'))
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                    stopwords_english = set(stopwords.words('english'))

                # Optional: Add stemmer if needed
                # stemmer = PorterStemmer()

                def clean_text(text):
                    text = text.lower()
                    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
                    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
                    # Optional: Stemming or Lemmatization could be added here
                    # text = ' '.join([stemmer.stem(word) for word in text.split()])
                    return text

                df_classification['cleaned_description'] = df_classification['description'].apply(clean_text)

                # 11.9. Initialize TF-IDF vectorizer for issue description.
                # Limit features to avoid overfitting and manage memory
                tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(stopwords_english))

                # 11.10. Fit and transform the cleaned descriptions.
                X_classification_tfidf = tfidf_vectorizer.fit_transform(df_classification['cleaned_description'])

                # 11.11. Assign the target variable (numerical labels)
                y_classification_numerical = y_classification.map(department_mapping)

                # 11.12. Split the data for category classification.
                X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
                    X_classification_tfidf, y_classification_numerical, test_size=0.2, random_state=42, stratify=y_classification_numerical
                )

                # 11.13. Model training (using Multinomial Naive Bayes for text classification)
                text_classifier = MultinomialNB()
                text_classifier.fit(X_train_cls, y_train_cls)

                # 11.14. Evaluate and display the issue categorization model.
                st.write("### Issue Categorization Model Evaluation")
                y_pred_cls = text_classifier.predict(X_test_cls)
                accuracy_classification = accuracy_score(y_test_cls, y_pred_cls)
                st.write(f"**Accuracy:** {accuracy_classification:.2f}")
                st.write("**Classification Report:**")
                try:
                    st.text(classification_report(y_test_cls, y_pred_cls, target_names=unique_departments))
                except ValueError:
                    st.text("Could not generate classification report (likely due to only one class present in predictions).")

                # 11.15. Interpret issue categorization results for non-technical people.
                st.write("### Understanding Issue Categorization Results")
                st.write("""
                The model categorizes issues based on the text descriptions provided by citizens.
                The accuracy indicates how often the model correctly predicts the department responsible for resolving the issue.
                A higher accuracy means the model is good at understanding the issue and routing it to the correct department, speeding up the resolution process.
                The classification report provides a more detailed breakdown of performance for each department, showing precision and recall.
                """)
            else:
                st.warning("No valid data available for issue categorization after cleaning.")
        else:
            st.warning("Required columns ('description', 'assigned_department') for issue categorization are missing in the uploaded dataset.")


        # --- Dashboard Insights ---
        st.subheader("5. Dashboard Insights")

        # 6. Escalation rate by assigned department
        st.write("### Escalation Rate by Assigned Department")
        if 'assigned_department' in df.columns and 'escalated' in df.columns:
            escalation_by_dept = df.groupby('assigned_department')['escalated'].mean().sort_values(ascending=False) * 100
            if not escalation_by_dept.empty:
                plt.figure(figsize=(7, 3))
                sns.barplot(x=escalation_by_dept.index, y=escalation_by_dept.values, palette='viridis')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel("Assigned Department")
                plt.ylabel("Escalation Rate (%)")
                plt.title("Escalation Rate by Assigned Department")
                st.pyplot(plt.gcf())
            else:
                st.warning("Could not calculate escalation rate by department. Ensure 'assigned_department' and 'escalated' columns are present and valid.")
        else:
            st.warning("Required columns ('assigned_department', 'escalated') are missing for department-based escalation analysis.")

        
        # 8. Impact of resolution time on escalation
        st.write("### Impact of Resolution Time on Escalation")
        if 'resolution_time' in df.columns and 'escalated' in df.columns:
            # Filter out potential outliers or very large resolution times if necessary for clarity
            df_viz = df[df['resolution_time'] < df['resolution_time'].quantile(0.95)].copy() # Consider top 5% for outlier removal
            avg_resolution_time_by_escalation = df_viz.groupby('escalated')['resolution_time'].mean()
            avg_resolution_time_by_escalation.index = avg_resolution_time_by_escalation.index.map({0: 'Not Escalated', 1: 'Escalated'})

            plt.figure(figsize=(7, 5))
            sns.barplot(x=avg_resolution_time_by_escalation.index, y=avg_resolution_time_by_escalation.values, palette='magma')
            plt.xlabel("Escalation Status")
            plt.ylabel("Average Resolution Time (days)")
            plt.title("Average Resolution Time vs. Escalation Status")
            st.pyplot(plt.gcf())
        else:
            st.warning("Required columns ('resolution_time', 'escalated') are missing for resolution time analysis.")


        # Example for Issue Volume Trends (Daily)
        st.write("### Daily Issue Submission Trend")
        if 'date_reported' in df.columns:
            # Ensure date_reported is in datetime format and handle NaT values
            df['date_reported_valid'] = pd.to_datetime(df['date_reported'], errors='coerce')
            daily_issue_counts = df['date_reported_valid'].value_counts().sort_index()
            # Filter out NaT values if any were created
            daily_issue_counts = daily_issue_counts[daily_issue_counts.index.notna()]

            if not daily_issue_counts.empty:
                plt.figure(figsize=(12, 6))
                daily_issue_counts.plot(kind='line')
                plt.xlabel("Date Reported")
                plt.ylabel("Number of Issues")
                plt.title("Daily Issue Submissions")
                plt.grid(True)
                st.pyplot(plt.gcf())
            else:
                st.warning("No valid daily issue submission data found.")
        else:
            st.warning("The 'date_reported' column is not available for daily trend analysis.")

        # --- Predict Escalation for a New Issue (Interactive Input) ---
        st.subheader("3. Predict Escalation for a New Issue")
        st.write("Enter the details of a new issue below to predict its escalation likelihood.")

        # Sidebar for user input for prediction
        with st.sidebar.header("Predict Escalation"):
            st.markdown("Enter issue details for prediction:")

            # Collect user input for features, ensuring they match the trained model's features
            # Use unique values from the uploaded dataset for selectboxes
            user_assigned_department_pred = st.sidebar.selectbox("Assigned Department", df['assigned_department'].unique() if 'assigned_department' in df.columns else ["Unknown"])
            user_assigned_level_pred = st.sidebar.selectbox("Assigned Level", df['assigned_level'].unique() if 'assigned_level' in df.columns else ["Unknown"])
            user_resolution_time_pred = st.sidebar.number_input("Resolution Time (days)", min_value=0, value=10, step=1)
            user_category_pred = st.sidebar.selectbox("Category", df['category'].unique() if 'category' in df.columns else ["Unknown"])

            # Create a DataFrame for the new issue
            new_issue_data_pred = pd.DataFrame({
                'assigned_department': [user_assigned_department_pred],
                'assigned_level': [user_assigned_level_pred],
                'resolution_time': [user_resolution_time_pred],
                'category': [user_category_pred]
            })

            # Predict escalation probability
            if st.sidebar.button("Predict Escalation"):
                try:
                    # Ensure the columns match what the model expects, even if not present in the uploaded df
                    # This is handled by the preprocessor in the pipeline
                    predicted_escalation_proba = model_pipeline_escalation.predict_proba(new_issue_data_pred)[0][1]
                    st.sidebar.write(f"**Likelihood of Escalation:** {predicted_escalation_proba:.2%}")
                    if predicted_escalation_proba > 0.5: # Default threshold
                        st.sidebar.warning("This issue is likely to be escalated.")
                    else:
                        st.sidebar.info("This issue is unlikely to be escalated.")
                except AttributeError:
                    st.sidebar.error("Escalation model is not trained. Please ensure data is loaded correctly.")
                except Exception as e:
                    st.sidebar.error(f"An error occurred during prediction: {e}")


        # --- Predict Department for a New Issue Description (Interactive Input) ---
        st.subheader("4. Predict Department for a New Issue Description")
        st.write("Enter a new issue description below to predict the appropriate department for routing.")

        with st.sidebar.header("Predict Department"):
            user_issue_description_cls = st.sidebar.text_area("Enter New Issue Description:", height=150)

            if st.sidebar.button("Categorize Issue"):
                if user_issue_description_cls:
                    try:
                        # Clean the new description
                        cleaned_user_description = clean_text(user_issue_description_cls)
                        # Vectorize the description
                        user_description_tfidf = tfidf_vectorizer.transform([cleaned_user_description])
                        # Predict the department index
                        predicted_dept_index = text_classifier.predict(user_description_tfidf)[0]
                        # Map back to department name
                        predicted_department = inverse_department_mapping[predicted_dept_index]
                        st.sidebar.success(f"The issue is categorized as: **{predicted_department}**")
                    except NameError:
                        st.sidebar.error("Issue categorization model is not trained. Please ensure data is loaded correctly.")
                    except Exception as e:
                        st.sidebar.error(f"An error occurred during categorization: {e}")
                else:
                    st.sidebar.warning("Please enter an issue description to categorize.")

    except Exception as e:
        st.error(f"An error occurred while processing the uploaded file: {e}")
        st.error("Please ensure the CSV file is correctly formatted and contains the expected columns.")

else:
    st.info("Please upload a CSV file using the sidebar to begin.")