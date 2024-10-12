<h1>Machine-Learning-Based-Diabetes-Outcome-Prediction</h1>

<h2>1.0 Business Problem</h2>
<p>
        Diabetes is a chronic disease that significantly impacts the health and lifestyle of individuals. Early detection and prediction of diabetes can help mitigate long-term complications. This project aims to predict whether an individual is likely to develop diabetes based on medical and demographic factors.
</p>

<h3>Problem Statement</h3>
<p>
        We are utilizing the <strong>PIMA Indian Diabetes Dataset</strong> to build machine learning models that can predict the likelihood of a patient being diabetic. The company’s business model depends on accurate predictions to provide value in healthcare decision-making.
</p>

<h3>Business Objectives</h3>
<ul>
        <li>Accurately predict diabetes in individuals.</li>
        <li>Minimize false negatives, as failing to detect diabetes could lead to serious health complications.</li>
        <li>Balance between precision and recall to maximize the efficiency of the model.</li>
</ul>

<h2>2.0 Business Assumptions</h2>
<ul>
        <li>Diabetes incidence is strongly linked to factors such as <strong>glucose levels</strong>, <strong>BMI</strong>, <strong>age</strong>, and <strong>pregnancy history</strong>.</li>
        <li>Early detection of diabetes can improve patient outcomes and reduce long-term healthcare costs.</li>
        <li>Machine learning models can be trained on historical medical data to predict the probability of diabetes in future patients.</li>
</ul>

<h2>3.0 Solution Strategy</h2>
<p>This project follows a structured data science approach to develop a predictive model for diabetes:</p>

<h3>Step 01: Data Collection</h3>
<p>We started by collecting and understanding the dataset.</p>

<p>The dataset contains medical records for <strong>768</strong> patients, with the following features:</p>
<ul>
    <li><strong>Pregnancies:</strong> Number of times pregnant</li>
    <li><strong>Glucose:</strong> Plasma glucose concentration</li>
    <li><strong>BloodPressure:</strong> Diastolic blood pressure</li>
    <li><strong>SkinThickness:</strong> Triceps skinfold thickness</li>
    <li><strong>Insulin:</strong> 2-hour serum insulin</li>
    <li><strong>BMI:</strong> Body mass index (weight in kg/(height in m)<sup>2</sup>)</li>
    <li><strong>DiabetesPedigreeFunction:</strong> A function that calculates genetic risk</li>
    <li><strong>Age:</strong> Age in years</li>
    <li><strong>Outcome:</strong> Binary variable (1 = diabetic, 0 = non-diabetic)</li>
</ul>

<h3>Step 02: Exploratory Data Analysis (EDA)</h3>
<p>EDA was performed to identify important trends and relationships in the data:</p>
<ul>
        <li> Descriptive statistics were applied to examine the distribution of the variables, and observe the target class distribution.</li>
        <li><strong>Correlation Matrix</strong>: Identified key features such as <strong>Glucose</strong>, <strong>BMI</strong>, and <strong>Age</strong> as strong predictors of diabetes.</li>
        <li><strong>Pair Plots</strong>: Visualized relationships between key variables for diabetic and non-diabetic groups.</li>
</ul>

<h3>Step 03: Data Preprocessing</h3>
<p>
        <strong>Handling Missing Data</strong>: Zero values in critical columns such as <strong>Glucose</strong>, <strong>BloodPressure</strong>, <strong>BMI</strong>, and <strong>Insulin</strong> were treated as missing data and replaced with median values.<br>
        <strong>No Feature Scaling</strong>: Based on the insights from EDA, Feature scaling was not applied, as most of the models used (such as Random Forest and Gradient Boosting) are not sensitive to feature scales.
</p>
<h3>Step 04: Feature Engineering</h3>
<p>No additional features were created in this phase. Existing features were standardized and scaled.</p>

<h3>Step 05: Model Building</h3>
<p>We trained and evaluated several machine learning models:</p>
<ul>
<li>Logistic Regression</li>
<li>Support Vector Machine (SVM)</li>
<li>Random Forest Classifier</li>
<li>Gradient Boosting Classifier</li>
</ul>

<h3>Step 06: Model Evaluation</h3>
<thead>
<tr>
<th>Algorithm</th>
<th>Accuracy</th>
 <th>Recall</th>
<th>F1 Score</th>
</tr>
</thead>
<tbody>
<tr>
<td>Logistic Regression</td>
<td>0.75974</td>
<td>0.618182</td>
<td>0.647619</td>
</tr>
<tr>
<td>Support Vector Machine</td>
<td>0.75974</td>
<td>0.545455</td>
<td>0.618557</td>
</tr>
<tr>
<td>Random Forest Classifier</td>
<td>0.75974</td>
<td>0.654545</td>
<td>0.660550</td>
</tr>
<tr>
                <td>Gradient Boosting Classifier</td>
                <td>0.73377</td>
                <td>0.709091</td>
                <td>0.655462</td>
</tr>
</tbody>
</table>

<h2>4.0 Data Insights</h2>
<ul>
        <li><strong>High Correlation Between Glucose and Diabetes</strong>: Higher glucose levels were significantly associated with diabetes, confirming existing medical knowledge.</li>
        <li><strong>Age and Diabetes Risk</strong>: Patients aged 50 and above showed a higher likelihood of being diabetic.</li>
        <li><strong>BMI and Diabetes</strong>: Patients with higher BMI were more likely to be diabetic.</li>
</ul>

<h2>5.0 Machine Learning Models Applied</h2>
<p>Here are the cross-validation results for the machine learning models:</p>

<table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Logistic Regression</td>
                <td>0.75974</td>
                <td>0.75</td>
                <td>0.62</td>
                <td>0.65</td>
            </tr>
            <tr>
                <td>Support Vector Machine</td>
                <td>0.75974</td>
                <td>0.76</td>
                <td>0.55</td>
                <td>0.62</td>
            </tr>
            <tr>
                <td>Random Forest Classifier</td>
                <td>0.75974</td>
                <td>0.75</td>
                <td>0.65</td>
                <td>0.66</td>
            </tr>
            <tr>
                <td>Gradient Boosting Classifier</td>
                <td>0.73377</td>
                <td>0.71</td>
                <td>0.71</td>
                <td>0.66</td>
            </tr>
        </tbody>
</table>
<h2>Observations:</h2>

<h3>Accuracy:</h3>
<ul>
    <li>All models except Gradient Boosting have the same accuracy of <strong>0.75974</strong>.</li>
    <li>Gradient Boosting has a slightly lower accuracy of <strong>0.733766</strong>.</li>
</ul>

<h3>Recall:</h3>
<ul>
    <li>Gradient Boosting has the highest recall (<strong>0.7091</strong>), meaning it is the best at identifying diabetic cases.</li>
    <li>Random Forest follows with a recall of <strong>0.6545</strong>, while Logistic Regression and SVM have lower recalls of <strong>0.618182</strong> and <strong>0.545455</strong>, respectively.</li>
</ul>

<h3>F1 Score:</h3>
<ul>
    <li>Random Forest has the highest F1 score (<strong>0.66055</strong>), followed closely by Gradient Boosting (<strong>0.655462</strong>).</li>
    <li>Logistic Regression has an F1 score of <strong>0.647619</strong>, while SVM has the lowest at <strong>0.618557</strong>.</li>
</ul>

<h2>Best Model:</h2>
<p><strong>Best Model:</strong> Random Forest Classifier</p>

<h3>Reasons:</h3>
<ul>
    <li><strong>Balanced Performance:</strong> Random Forest offers a good balance between accuracy, recall, and F1 score. It has the same accuracy as Logistic Regression and SVM but outperforms them in terms of recall and F1 score.</li>
    <li><strong>High F1 Score:</strong> The F1 score is crucial as it combines precision and recall into a single metric. Random Forest's score indicates it is effectively balancing both metrics.</li>
    <li><strong>Robustness:</strong> Random Forest is known for its robustness to overfitting, especially in datasets with a higher number of features, which can be advantageous in real-world scenarios.</li>
</ul>

<h3>Alternative Consideration:</h3>
<p>Gradient Boosting could also be considered a strong contender due to its high recall, meaning it is excellent at identifying diabetic cases. However, it sacrifices some overall accuracy compared to Random Forest.</p>

<h2>Conclusion:</h2>
<p>In conclusion, while Gradient Boosting excels in recall, Random Forest is recommended as the best model due to its well-rounded performance across all metrics. The choice ultimately depends on whether the priority is on minimizing false negatives (favoring Gradient Boosting) or achieving a more balanced performance (favoring Random Forest).</p>

<h2>6.0 Business Results</h2>
<p>
        Using the <strong>Random Forest Classifier</strong> model:
</p>
<ul>
        <li>The model can detect <strong>65.5%</strong> of diabetic cases with high precision and a balanced accuracy of <strong>75.97%</strong>.</li>
        <li>This performance ensures that a reliable number of patients at risk are identified, which can help in early medical interventions.</li>
</ul>

<h2>7.0 Conclusions</h2>
<p>
        This project demonstrated the power of machine learning in predicting diabetes based on demographic and medical data. The <strong>Random Forest</strong> model achieved the best balance between accuracy and recall, making it suitable for medical decision-making.
</p>

<h2>8.0 Lessons Learned</h2>
<ul>
        <li><strong>Handling missing data</strong> is crucial for improving model performance.</li>
        <li><strong>A balance between precision and recall</strong> should always be maintained, especially in medical applications where false negatives can have severe consequences.</li>
</ul>

<h2>9.0 Next Steps</h2>
<ul>
        <li><strong>Hyperparameter Tuning</strong>: Further tuning of the selected model to optimize performance.</li>
        <li><strong>Feature Engineering</strong>: Introduce new medical features and domain-specific knowledge to improve the prediction.</li>
        <li><strong>Model Deployment</strong>: Deploy the model as an API for real-time predictions.</li>
</ul>


<h2>Dataset</h2>
<p>The PIMA Indian Diabetes Dataset used in this analysis is available on the UCI Machine Learning Repository. It consists of medical records of female patients and contains the following attributes:</p>
<ul>
    <li>Pregnancies</li>
    <li>Glucose</li>
    <li>Blood Pressure</li>
    <li>Skin Thickness</li>
    <li>Insulin</li>
    <li>BMI</li>
    <li>Diabetes Pedigree Function</li>
    <li>Age</li>
    <li>Outcome (Diabetes Yes/No)</li>
</ul>

<h2>Libraries Used</h2>
<p>The following Python libraries were used in this project:</p>
<ul>
    <li><strong>Pandas:</strong> For data manipulation and analysis.</li>
    <li><strong>Numpy:</strong> For numerical computing and handling arrays.</li>
    <li><strong>Scikit-learn:</strong> For implementing machine learning models and evaluation metrics.</li>
    <li><strong>Matplotlib:</strong> For data visualization.</li>
    <li><strong>Seaborn:</strong> For advanced statistical graphics.</li>
</ul>
