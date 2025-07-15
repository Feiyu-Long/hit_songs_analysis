# import relevant modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def display_choice():
    # Description: display options to users
    # Parameter: None
    # Return: None
    print("Welcome. This program allows you to access a database on hit songs in recent years.\nSelect one operation from the following:")
    print("A. Sort DataFrame")
    print("B. Search for object in column")
    print("C. Filter by release date")
    print("D. Filter by explicit content")
    print("E. Predict artist future popularity")
    print("F. Save current DataFrame as image")
    print("G. Reset all data operations")
    print("H. Exit program")

def get_header(file_name):
    # Description: print a list in a more reader-friendly format.
    # Parameter: 1
    # 1. file_name: str storing the name of the csv file
    # Return: header_list storing a list of headers
    with open(file_name, "r") as fileObj:
        header_line = fileObj.readline()
        header_list = header_line.strip().split("	")
    return header_list

def pretty_print_list(list_name):
    # Description: print a list in a more reader-friendly format.
    # Parameter: 1
    # 1. list_name storing a list of countries and regions
    # Return: None
    n=0
    for item in list_name:
        n+=1
        print(str(n)+". "+item)

def data_wrangling(file_name):
    # Description: store all data in a dataFrame
    # Parameter: 1
    # 1. file_name str storing the name of the csv file
    # Return: wrangled dataFrame

    # read file and create dataFrame
    df = pd.read_csv(file_name,delimiter="\t")

    # drop duplicates and NA
    df.drop_duplicates(inplace=True)
    df.dropna(axis=0,inplace=True)

    # Clean up the 'name_artists' column
    def clean_artists(value):
        try:
            artist_list = ast.literal_eval(value)
            return ", ".join(artist_list)
        except:
            return value

    df["name_artists"] = df["name_artists"].apply(clean_artists)

    return df

def object_sort(df,column_num,sort_num):
    # Description: allow user to sort dataFrame based on select column and display sorted data
    # Parameter: 3
    # 1. df storing the input dataFrame
    # 2. column_num storing an integer representing the index number corresponding to the select column
    # 3. sort_num storing an integer representing either ascending (0) or descending (1)

    # correspond column_num to column_name
    col_name = df.columns[column_num]

    # ensure the column is sortable
    if not pd.api.types.is_numeric_dtype(df[col_name]) and not pd.api.types.is_string_dtype(df[col_name]):
        print("Selected column is not sortable.")
        return df

    # determine sorting order: 0 = ascending, 1 = descending
    ascending = True if sort_num == 0 else False

    # sort and display
    df_sorted = df.sort_values(by=col_name, ascending=ascending)
    return df_sorted


def object_search(df,column_num,object_name):
    # Description: allow user to search for the occurrence of a data point in select column
    # Parameter: 3
    # 1. df storing the input dataFrame
    # 2. column_num storing an integer representing the index number corresponding to the select column
    # 3. object_name storing the inputted data point request by user
    # Return: filtered dataFrame

    col_name = df.columns[column_num]
    df_filtered = df[df[col_name] == object_name]
    if df_filtered.empty:
        print("Search complete. No match found.")
        return df
    return df_filtered

def time_search(df,year_num=None,month_num=None,date_num=None):
    # Description: allow user to search based on its release date components
    # Parameter: 4
    # 1. df storing the input dataFrame
    # 2. year_num storing an integer representing the release year, set default as None
    # 3. month_num storing an integer representing the release month, set default as None
    # 4. date_num storing an integer representing the release day, set default as None
    # Return: wrangled dataFrame

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Apply filters if provided
    if year_num is not None:
        df = df[df['release_date'].dt.year == year_num]
    if month_num is not None:
        df = df[df['release_date'].dt.month == month_num]
    if date_num is not None:
        df = df[df['release_date'].dt.day == date_num]

    df_filtered_2 = df.drop(columns=["song_id", "id_artists"])
    return df
# sort df in time order

def filter_explicit(df,filter_num):
    # Description: filter by explicit or non-explicit
    # Parameter: 2
    # 1. df storing the input dataFrame
    # 2. filter_num determining whether filter by explicit or non-explicit
    # Return: filtered dataFrame
    if filter_num == 0:
        df_filtered = df[df["explicit"] == False]
    else:
        df_filtered = df[df["explicit"] == True]
    return df_filtered

def artist_popularity_prediction(df,artist_name,future_hits_num):
    # Description: visualize datapoints of artists' hits and predict the future popularity of their hits using ML
    # Parameter: 3
    # 1. df storing the input dataFrame
    # 2. artist_name storing the name of the artist
    # 3. future_hits_num storing the number of future hits required to predict
    # Return: None
    df_artist = object_search(df, 3, artist_name).copy()
    if df_artist.empty:
        print("No songs found for this artist.")

    df_artist['release_date'] = pd.to_datetime(df_artist['release_date'], errors='coerce')
    df_artist = df_artist.dropna(subset=['release_date', 'popularity'])

    if len(df_artist) < 5:
        print("Not enough data points to train.")
        return df_artist

    X = np.array(df_artist['release_date'].map(lambda x: x.toordinal())).reshape(-1, 1)
    y = df_artist['popularity'].values

    # Fit polynomial regression
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)

    last_date = df_artist['release_date'].max()
    future_dates = [last_date + pd.Timedelta(days=30 * i) for i in range(1, future_hits_num + 1)]
    X_future = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    y_pred = model.predict(X_future)

    dense_dates = pd.date_range(df_artist['release_date'].min(), future_dates[-1], freq='7D')
    X_dense = np.array([d.toordinal() for d in dense_dates]).reshape(-1, 1)
    y_dense_pred = model.predict(X_dense)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df_artist['release_date'], y, color='blue', label='Historical')
    plt.plot(dense_dates,y_dense_pred, color='purple', linestyle='-.', alpha=0.6, label='Fitted + Future Curve')
    plt.scatter(future_dates, y_pred, color='red',label='Future Predictions')
    plt.xlabel("Release Date")
    plt.ylabel("Popularity")
    plt.title("Popularity Prediction for "+artist_name+" (Poly Regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predicted_popularity_for_"+artist_name+".jpg", dpi=300)
    plt.close()

    df_future = pd.DataFrame({
        'predicted_release_date': future_dates,
        'predicted_popularity': y_pred
    })
    print(df_future.head())

def save_df_as_image(df, file_name, max_rows):
    # Description: save input dataFrame as an image
    # Parameter: 3
    # 1. df storing the input dataFrame
    # 2. file_name storing the name of the image file
    # 3. max_rows storing the top number of rows included in the saved image
    # Return: None

    # Limit number of rows shown
    if df is None or df.empty:
        print("DataFrame is empty — nothing to save.")
        return

    # limit rows included in the plot
    df_to_plot = df.head(max_rows)

    # create figure
    fig, ax = plt.subplots(figsize=(20, 0.5 * len(df_to_plot)))  # height scales with rows
    ax.axis('tight')
    ax.axis('off')

    # create, customize and save figure
    table = ax.table(cellText=df_to_plot.values,
                     colLabels=df_to_plot.columns,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    fig.tight_layout()
    fig.savefig(file_name+".jpg", dpi=300)
# descriptive characteristics score across a certain time period: visualize


def main():
    file_name="hits_dataset.csv"
    headers=get_header(file_name)
    df=data_wrangling(file_name)
    current_df=df.copy()
    # set plot style
    plt.style.use("ggplot")

    choice = ""
    while choice != "H":
        display_choice()
        choice = input("Enter your choice (A-G, H to exit): ").strip().upper()
        if choice == "A":
            print("Choose one column to sort:")
            pretty_print_list(headers)
            column_num_raw = input("Select the column number to sort by: ")
            while column_num_raw.isdigit() == False or int(column_num_raw) not in range(len(headers)+1):
                column_num_raw = input("Please enter a valid input: ")
            column_num=int(column_num_raw)-1
            sort_order = input("Enter 0 for ascending or 1 for descending: ")
            while sort_order.isdigit() == False or int(sort_order) not in range(2):
                sort_order = input("Please enter a valid input: ")
            sort_order=int(sort_order)
            current_df = object_sort(current_df, column_num, sort_order)
            print("Sorting complete.")

        elif choice == "B":
            print("Choose one column to search:")
            pretty_print_list(list(current_df.columns))
            column_num_raw = input("Select the column number to sort by: ")
            while column_num_raw.isdigit() == False or int(column_num_raw) not in range(len(headers) + 1):
                column_num_raw = input("Please enter a valid input:")
            column_num = int(column_num_raw) - 1
            search_value = input("Enter the value to search for: ")
            if column_num == 3:
                filtered_df=current_df[current_df["name_artists"].str.contains(search_value, case=False, na=False)].copy()
            else:
                filtered_df = object_search(current_df, column_num, search_value)
            if not filtered_df.empty:
                current_df = filtered_df  # update df view
                print("Searching complete.")

        elif choice == "C":
            y = input("Enter year (or leave blank): ")
            while y.isdigit() == False:
                y=input("Please enter a valid input: ")
            m = input("Enter month (or leave blank): ")
            while m.isdigit() == False:
                m=input("Please enter a valid input: ")
            d = input("Enter day (or leave blank): ")
            while d.isdigit() == False:
                d=input("Please enter a valid input: ")
            year_num = int(y) if y.strip() else None
            month_num = int(m) if m.strip() else None
            date_num = int(d) if d.strip() else None
            current_df = time_search(current_df, year_num, month_num, date_num)
            print("Searching complete.")

        elif choice == "D":
            filter_num = input("Enter 0 to filter for non-explicit, 1 to filter for explicit: ")
            while filter_num.isdigit()==False or int(filter_num) not in range(2):
                filter_num=input("Please enter a valid input: ")
            filter_num=int(filter_num)
            current_df = filter_explicit(current_df, filter_num)
            print("Filtering complete.")

        elif choice == "E":
            artist_name_input = input("Enter artist name (e.g. Ariana Grande): ").strip()
            future_hits_num = input("Enter the number of future hits to predict: ")
            while future_hits_num.isdigit() == False:
                future_hits_num=input("Please enter a valid input: ")
            future_hits_num=int(future_hits_num)
            df_artist = current_df[current_df["name_artists"].str.contains(artist_name_input, case=False, na=False)].copy()
            if df_artist.empty:
                print("No songs found for this artist.")
            else:
                artist_popularity_prediction(df_artist, artist_name_input, future_hits_num)

        elif choice == "F":
            file_name = input("Enter file name to save current DataFrame as image: ")
            max_rows = input("Enter maximum rows to include in the image: ")
            while max_rows.isdigit() == False:
                max_rows=input("Please enter a valid input: ")
            max_rows=int(max_rows)
            save_df_as_image(current_df.drop(columns=["song_id", "id_artists"]), file_name, max_rows)
            print("Current DataFrame view saved.")

        elif choice == "G":
            # reset df
            current_df=df.copy()

        elif choice == "H":
            print("\nProgram exited.")

        else:
            print("\nPlease select a valid choice (A - H).")

main()


