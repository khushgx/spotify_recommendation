Spotify Playlist Recommendation Algorithm

Initially, we were planning on doing a more empirical project that focused on analyzing Twitter’s open-source recommendation algorithm.
However, we realized that much of the codebase was written in Scala, which was a language that no one in our group was familiar with.
Therefore, we decided to switch to an implementation-based project, using the Spotify and Genius APIs to construct a recommendation algorithm that can suggest public playlists for a user to try based on their existing personal playlists.
Our program takes in a user’s Spotify ID and prompts them to select a genre that they would like to explore.
Based on the genre they select, our algorithm retrieves the top public playlists in that genre and compares it to the user’s existing personal playlists in terms of both audio and lyrical similarity.
It then uses the PageRank algorithm to suggest the top five most similar playlists for the user to try.

For our project, we used graphs and graph algorithms.
We represented each of the playlists as a node in our graph and drew edges between every pair of vertices to create a complete graph.
We weighted the edges from 0 to 1 to represent the similarity between two playlists, where 0 represents no similarity and 1 represents complete similarity.
After creating this graph, we ran the PageRank algorithm on it to determine the five most important nodes in the graph and recommended those to the user.

We also used Document Search principles when calculating the lyric similarity scores between playlists.
We transformed the lyrics in each song into Tf-idf vectors in order to ensure that all songs were represented by vectors of the same length.
From there, we calculated the cosine similarity between every pair of vectors in two playlists.
We then took the average of these results in order to get our final similarity score between two playlists.

Finally, our project makes use of advanced topics because it is a recommendation algorithm.
By combining audio similarity scores with lyric similarity scores, we were able to calculate an overall similarity score between two playlists.
We then created a graph and ran PageRank on it in order to create a comprehensive recommendation system that can suggest public playlists to users based on the similarity scores to their own playlists.

In terms of our work breakdown, Khush worked on the API calls with Spotify and Genius.
Daniel worked on the front-end and integrated it to the back end.
Sahishnu worked on the PageRank algorithm and calculating the similarity scores.
All three of us debugged our code to some extent and helped bring our separate components together to create a cohesive recommendation algorithm.
