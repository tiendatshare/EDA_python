import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# add title for the app
st.title("Exploratory Data Analysis")

# sidebar title
st.sidebar.title("Upload Data")

# add tabs
tab1, tab2, tab3, tab4,tab5,tab6, tab7 = st.tabs(["Data Info", "Numeric Features", "Categorical Features","Numeric & Categorical","Numeric & Numeric","Categorical & Categorical","Model"])

# add file-uploader widget in sidebar
uploaded_data = st.sidebar.file_uploader("Choose a CSV file")

@st.cache_data
def load_data(file_name):
  # read CSV file
  data = pd.read_csv(file_name)
  return data

if uploaded_data is not None:
  # read csv
  df = load_data(uploaded_data)

with tab1:
  if uploaded_data is not None:
    # extract meta-data from the uploaded dataset
    st.header("Meta-data")

    row_count = df.shape[0]

    column_count = df.shape[1]
    
    # Use the duplicated() function to identify duplicate rows
    duplicates = df[df.duplicated()]
    duplicate_row_count =  duplicates.shape[0]

    missing_value_row_count = df[df.isna().any(axis=1)].shape[0]

    table_markdown = f"""
      | Description | Value | 
      |---|---|
      | Number of Rows | {row_count} |
      | Number of Columns | {column_count} |
      | Number of Duplicated Rows | {duplicate_row_count} |
      | Number of Rows with Missing Values | {missing_value_row_count} |
      """

    st.markdown(table_markdown)

    st.header("Columns Type")

    # get feature names
    columns = list(df.columns)

    # create dataframe
    column_info_table = pd.DataFrame({
          "column": columns,
          "data_type": df.dtypes.tolist()
    })
    
    # display pandas dataframe as a table
    st.dataframe(column_info_table, hide_index=True)

with tab2:
  if uploaded_data is not None:
    # find numeric features  in the dataframe
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # add selection-box widget
    selected_num_col = st.selectbox("Which numeric column do you want to explore?", numeric_cols)

    st.header(f"{selected_num_col} - Statistics")
    
    col_info = {}
    col_info["Number of Unique Values"] = len(df[selected_num_col].unique())
    col_info["Number of Rows with Missing Values"] = df[selected_num_col].isnull().sum()
    col_info["Number of Rows with 0"] = df[selected_num_col].eq(0).sum()
    col_info["Number of Rows with Negative Values"] = df[selected_num_col].lt(0).sum()
    col_info["Average Value"] = df[selected_num_col].mean()
    col_info["Standard Deviation Value"] = df[selected_num_col].std()
    col_info["Minimum Value"] = df[selected_num_col].min()
    col_info["Maximum Value"] = df[selected_num_col].max()
    col_info["Median Value"] = df[selected_num_col].median()

    info_df = pd.DataFrame(list(col_info.items()), columns=['Description', 'Value'])
    # display dataframe as a markdown table
    st.dataframe(info_df)

    st.header("Histogram")
    fig = px.histogram(df, x=selected_num_col)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if uploaded_data is not None:
        # find categorical columns in the dataframe
        cat_cols = df.select_dtypes(include='object')
        cat_cols_names = cat_cols.columns.tolist()
 
        # add select widget
        selected_cat_col = st.selectbox("Which text column do you want to explore?", cat_cols_names)
 
        st.header(f"{selected_cat_col}")
        
        # Tạo biểu đồ phân phối của các giá trị trong cột văn bản
        value_counts = df[selected_cat_col].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': 'Value', 'y': 'Count'})
        st.plotly_chart(fig)
        
        # add categorical column stats
        cat_col_info = {}
        cat_col_info["Number of Unique Values"] = len(df[selected_cat_col].unique())
        cat_col_info["Number of Rows with Missing Values"] = df[selected_cat_col].isnull().sum()
        cat_col_info["Number of Empty Rows"] = df[selected_cat_col].eq("").sum()
        cat_col_info["Number of Rows with Only Whitespace"] = len(df[selected_cat_col][df[selected_cat_col].str.isspace()])
        cat_col_info["Number of Rows with Only Lowercases"] = len(df[selected_cat_col][df[selected_cat_col].str.islower()])
        cat_col_info["Number of Rows with Only Uppercases"] = len(df[selected_cat_col][df[selected_cat_col].str.isupper()])
        cat_col_info["Number of Rows with Only Alphabet"] = len(df[selected_cat_col][df[selected_cat_col].str.isalpha()])
        cat_col_info["Number of Rows with Only Digits"] = len(df[selected_cat_col][df[selected_cat_col].str.isdigit()])
        cat_col_info["Mode Value"] = df[selected_cat_col].mode()[0]
 
        cat_info_df = pd.DataFrame(list(cat_col_info.items()), columns=['Description', 'Value'])
        st.dataframe(cat_info_df)

with tab4:
    if uploaded_data is not None:
        # find numeric and categorical columns in the dataframe
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        # add select widgets for numeric and categorical columns
        selected_numeric_col = st.selectbox("Select numeric column:", numeric_cols, key="numeric_selectbox")
        selected_cat_col = st.selectbox("Select categorical column:", cat_cols, key="categorical_selectbox")

        st.header(f"{selected_numeric_col} vs {selected_cat_col}")

        # Create box plot to visualize the relationship between numeric and categorical variables
        fig = px.box(df, x=selected_cat_col, y=selected_numeric_col)
        st.plotly_chart(fig, use_container_width=True)


#them 3 tab nưa, numberic&numberic - bieu do scatter, categori & categori- bieu do nhiet, model- chon model kieu phan loai, du doan,... với biến tụ chọn

with tab5:
    if uploaded_data is not None:
        # find numeric columns in the dataframe
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        # add select widgets for numeric columns
        selected_numeric_col1 = st.selectbox("Select first numeric column:", numeric_cols, key="numeric_selectbox1")
        selected_numeric_col2 = st.selectbox("Select second numeric column:", numeric_cols, key="numeric_selectbox2")

        st.header(f"{selected_numeric_col1} vs {selected_numeric_col2}")

        # Create scatter plot to visualize the relationship between two numeric variables
        fig = px.scatter(df, x=selected_numeric_col1, y=selected_numeric_col2)
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    if uploaded_data is not None:
        # find categorical columns in the dataframe
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        # add select widgets for categorical columns
        selected_cat_col1 = st.selectbox("Select first categorical column:", cat_cols, key="categorical_selectbox1")
        selected_cat_col2 = st.selectbox("Select second categorical column:", cat_cols, key="categorical_selectbox2")

        st.header(f"{selected_cat_col1} vs {selected_cat_col2}")

        # Create heatmap to visualize the relationship between two categorical variables
        crosstab_df = pd.crosstab(df[selected_cat_col1], df[selected_cat_col2])
        fig = px.imshow(crosstab_df, labels=dict(x=selected_cat_col2, y=selected_cat_col1), x=crosstab_df.columns, y=crosstab_df.index)
        st.plotly_chart(fig, use_container_width=True)

with tab7:
    if uploaded_data is not None:
        st.sidebar.title("Model Settings")
        model_type = st.sidebar.selectbox("Select Model Type:", ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])
        
        # find numeric columns in the dataframe
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        target_variable = st.sidebar.selectbox("Select Target Variable:", numeric_cols)

        # add select widget for independent variable
        independent_variable = st.sidebar.selectbox("Select Independent Variable:", numeric_cols)

        st.header("Model Prediction")

        X = df[[independent_variable]]
        y = df[target_variable]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize selected model
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        elif model_type == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Display model evaluation metrics
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        st.write(f"R-squared (R2): {r2}")

        # Plot predictions vs. actual values
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        st.pyplot(fig)