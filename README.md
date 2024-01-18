# Template for Machine Learning projects


The main objective of this project is to predict the possible success of a movie based on its information. In order to achieve this objective, the Machine Learning model used will be de K-nearest-neighbors, which classifies an entry according to its location.

Step 1: Loading the dataset
We must load the two files and store them in two separate data structures (Pandas DataFrames). On one side we will have stored the information of the movies and their credits.

Step 2: Creation of a database
Create a database to store the two DataFrames in separate tables. Then join the two tables with SQL (and integrate it with Python) to generate a third table containing information from both tables unified. The key through which the join can be done is the title of the movie (title).

Now, clean the generated table and leave only the following columns:

movie_id
title
overview
genres
keywords
cast
crew

Step 3: Transform the data
As you can see, there are some JSON formatted columns. Select, from each of the JSONs, select the name attribute and replace the genres and keywords columns. For the cast column, select the first three names.

The only columns left to modify are crew (team) and overview (summary). For the first column, convert it to contain the name of the director. For the second, convert it to a list.

Once we have finished processing the columns and the recommendation model is not confused, for example, between Jennifer Aniston and Jennifer Conelly, we will remove the spaces between the words. Apply this function to the columns genres, cast, crew and keywords.

Finally, we will reduce our dataset by combining all of our previous converted columns into a single column called tags (which we will create). This column will now have all the elements separated by commas and then we will replace them with blanks. It should look something like this:

new_df["tags"][0]

>>>>"In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron

Step 4: Build a KNN
To solve this problem we will create our own KNN. The first thing to do is to vectorize the text following the same steps you learned in the previous lesson.

Once you have done that, we would have to choose a distance to compare text. In this module we have seen a few, and the only one compatible with what we want to do is the cosine distance:

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

With this code we can see the similarity between our vectors (vector representations of the tags column).

Finally, we can design our similarity function based on the cosine distance. Our proposal is as follows:

def recommend(movie):
    movie_index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse = True , key = lambda x: x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)