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
    print("\nWelcome. This program allows you to access a database on hit songs in recent years.\nSelect one operation from the following:")
    print("A. Sort Data")
    print("B. Search for object in column")
    print("C. Filter by release date")
    print("D. Filter by explicit content")
    print("E. Visualize and predict values of a column from an artist")
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

    # correspond column name to df column
    col_name = df.columns[column_num]

    # apply filter
    df_filtered = df[df[col_name] == object_name]

    # return original df when df is empty
    if df_filtered.empty:
        print("Search complete. No match found.")
        return df

    # return filtered df
    return df_filtered

def time_search(df,year_num=None,month_num=None,date_num=None):
    # Description: allow user to search based on its release date components
    # Parameter: 4
    # 1. df storing the input dataFrame
    # 2. year_num storing an integer representing the release year, set default as None
    # 3. month_num storing an integer representing the release month, set default as None
    # 4. date_num storing an integer representing the release day, set default as None
    # Return: wrangled dataFrame

    # convert to datetime variable
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Apply filters if provided
    if year_num is not None:
        df = df[df['release_date'].dt.year == year_num]
    if month_num is not None:
        df = df[df['release_date'].dt.month == month_num]
    if date_num is not None:
        df = df[df['release_date'].dt.day == date_num]

    return df

def filter_explicit(df,filter_num):
    # Description: filter by explicit or non-explicit
    # Parameter: 2
    # 1. df storing the input dataFrame
    # 2. filter_num determining whether filter by explicit or non-explicit
    # Return: filtered dataFrame

    # use branching to apply filter
    if filter_num == 0:
        df_filtered = df[df["explicit"] == False]
    else:
        df_filtered = df[df["explicit"] == True]

    return df_filtered

def artist_value_prediction(df,artist_name,column_num, future_hits_num):
    # Description: visualize datapoints of artists' hits and predict future datapoints of the selected column of their hits using ML
    # Parameter: 3
    # 1. df storing the input dataFrame
    # 2. artist_name storing the name of the artist
    # 3. column_num storing the index of the input column
    # 4. future_hits_num storing the number of future hits required to predict
    # Return: None

    # ensure valid artist
    df_artist = object_search(df, 3, artist_name).copy()
    if df_artist.empty:
        print("No songs found for this artist.")

    # ensure column contains numeric value
    target_col = df_artist.columns[column_num]
    if not pd.api.types.is_numeric_dtype(df_artist[target_col]):
        print("The selected column "+target_col+" is not numeric.")
        return df_artist

    # convert dates to datetime
    df_artist['release_date'] = pd.to_datetime(df_artist['release_date'], errors='coerce')

    # drop na values
    df_artist = df_artist.dropna(subset=['release_date', target_col])

    # ensure enough datapoints for training
    if len(df_artist) < 3:
        print("Not enough data points to train.")
        return df_artist

    # set X and y subsets
    X = np.array(df_artist['release_date'].map(lambda x: x.toordinal())).reshape(-1, 1)
    y = df_artist[target_col].values

    # Fit polynomial regression
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)

    # determine future dates
    last_date = df_artist['release_date'].max()
    future_dates = [last_date + pd.Timedelta(days=30 * i) for i in range(1, future_hits_num + 1)]

    # predict based on future dates
    X_future = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    y_pred = model.predict(X_future)

    # apply dense dates to fit the line
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
    plt.title(str(target_col)+" prediction for "+artist_name+" (Poly Regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predicted_"+str(target_col)+"_for_"+artist_name+".jpg", dpi=300)
    plt.close()

    # wrap predicted dates and values into new df and display info
    df_future = pd.DataFrame({
        'predicted_release_date': future_dates,
        'predicted_'+str(target_col): y_pred
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

    # calculate figure size dynamically
    n_rows, n_cols = df_to_plot.shape
    cell_height = 0.5
    cell_width = 2.5  
    fig_height = max(1, cell_height * (n_rows + 1))  
    fig_width = max(6, cell_width * n_cols)

    # dynamic font size to fit in the graph
    if n_rows > 30 or n_cols > 8:
        font_size = 8
    elif n_rows > 15 or n_cols > 6:
        font_size = 9
    else:
        font_size = 10

    # create figure
    fig, ax = plt.subplots(figsize=(fig_width,fig_height)) 
    ax.axis("tight")
    ax.axis("off")

    # create, customize and save figure
    table = ax.table(cellText=df_to_plot.values,
                     colLabels=df_to_plot.columns,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.2, 1.2)

    # tight layout
    fig.tight_layout()

    # save figure
    fig.savefig(file_name+".jpg", dpi=300)

def main():
    # Description: main user interface
    # Parameters: None
    # Returns: None

    # read the file, created df, and wrangle df
    file_name="hits_dataset.csv"
    headers=get_header(file_name)
    df=data_wrangling(file_name)

    # create copy of original df
    current_df=df.copy()

    # set plot style
    plt.style.use("ggplot")

    choice = ""

    # use while loop to allow user to execute multiple demands
    while choice != "H":

        # display choice
        display_choice()

        # get choice
        choice = input("Enter your choice (A-G, H to exit): ").strip().upper()

        # Choice A: Sort Data
        if choice == "A":

            # display column choice
            print("Choose one column to sort:")
            pretty_print_list(headers)

            # get valid column number input
            column_num_raw = input("Select the column number to sort by: ")
            while column_num_raw.isdigit() == False or int(column_num_raw) not in range(len(headers)+1):
                column_num_raw = input("Please enter a valid input: ")
            column_num=int(column_num_raw)-1

            # get valid sort number input
            sort_order = input("Enter 0 for ascending or 1 for descending: ")
            while sort_order.isdigit() == False or int(sort_order) not in range(2):
                sort_order = input("Please enter a valid input: ")
            sort_order=int(sort_order)

            # sort df
            current_df = object_sort(current_df, column_num, sort_order)
            print("Sorting complete.")

        # Choice B: Search for values in a column
        elif choice == "B":

            # display column choice
            print("Choose one column to search:")
            pretty_print_list(list(current_df.columns))

            # get valid column number
            column_num_raw = input("Select the column number to sort by: ")
            while column_num_raw.isdigit() == False or int(column_num_raw) not in range(len(headers) + 1):
                column_num_raw = input("Please enter a valid input:")
            column_num = int(column_num_raw) - 1

            # get input value
            search_value = input("Enter the value to search for: ")

            # for artist name column, search for inclusion of the input value
            # in other column, apply object search and get filtered df
            if column_num == 3:
                filtered_df=current_df[current_df["name_artists"].str.contains(search_value, case=False, na=False)].copy()
            else:
                filtered_df = object_search(current_df, column_num, search_value)
            if not filtered_df.empty:
                current_df = filtered_df
                print("Searching complete.")

        # Choice C: Filter by release date
        elif choice == "C":

            # use while loops to get valid year, month and day inputs
            while True:
                y = input("Enter year (or leave blank): ").strip()
                if y == "":
                    year_num = None
                    break
                elif y.isdigit():
                    year_num = int(y)
                    break
                else:
                    print("Please enter a valid input.")

            while True:
                m = input("Enter month (or leave blank): ").strip()
                if m == "":
                    month_num = None
                    break
                elif m.isdigit():
                    month_num = int(m)
                    break
                else:
                    print("Please enter a valid input.")

            while True:
                d = input("Enter day (or leave blank): ").strip()
                if d == "":
                    day_num = None
                    break
                elif d.isdigit():
                    day_num = int(d)
                    break
                else:
                    print("Please enter a valid input.")

            # update current df
            current_df = time_search(current_df, year_num, month_num, day_num)
            print("Searching complete.")

        # Choice D: Filter by explicit content
        elif choice == "D":

            # get and ensure valid input for filter_num
            filter_num = input("Enter 0 to filter for non-explicit, 1 to filter for explicit: ")
            while filter_num.isdigit()==False or int(filter_num) not in range(2):
                filter_num=input("Please enter a valid input: ")
            filter_num=int(filter_num)

            # update current df with filter applied
            current_df = filter_explicit(current_df, filter_num)
            print("Filtering complete.")

        # Choice E: Visualize and predict artist hit songs and their corresponding selected column values
        elif choice == "E":

            # get input for artist name
            artist_name_input = input("Enter artist name (e.g. Ariana Grande): ").strip()

            # display column choice
            print("Choose a column of values:")
            pretty_print_list(headers)

            # get and ensure valid input for column
            column_num_raw = input("Select column number for value prediction: ")
            while column_num_raw.isdigit()==False or int(column_num_raw) not in range(len(headers)+1):
                column_num_raw=input("Please enter a valid input: ")
            column_num=int(column_num_raw)-1

            # get and ensure valid input for prediction datapoints
            future_hits_num = input("Enter the number of future hits to predict: ")
            while future_hits_num.isdigit() == False:
                future_hits_num=input("Please enter a valid input: ")
            future_hits_num=int(future_hits_num)

            # for artist name column, search for inclusion of the input value
            df_artist = df[df["name_artists"].str.contains(artist_name_input, case=False, na=False)].copy()

            # visualize and value prediction
            artist_value_prediction(df_artist, artist_name_input, column_num, future_hits_num)

        # Choice F: Save current dataframe to image
        elif choice == "F":

            # get file name
            file_name = input("Enter file name to save current DataFrame as image: ")

            # get valid input for max rows to display
            max_rows = input("Enter maximum rows to include in the image: ")
            while max_rows.isdigit() == False:
                max_rows=input("Please enter a valid input: ")
            max_rows=int(max_rows)

            # save dataframe as image
            save_df_as_image(current_df.drop(columns=["song_id", "id_artists"]), file_name, max_rows)
            print("Current DataFrame view saved.")

        elif choice == "G":
            # reset df
            current_df=df.copy()

        elif choice == "H":
            # display message and exit
            print("\nProgram exited.")

        else:
            # display message reset choice
            print("\nPlease select a valid choice (A - H).")

# call main
main()
