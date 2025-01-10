# Recommendation System and Trend Analysis on Spotify
A detailed analysis of music marketing through Big Data

## Problem Setting
Today, the overwhelming volume of diverse and complex music data far exceeds the basic needs and processing capacity of listeners, often resulting in information overload. High-end music charts aim to address this by analyzing various attributes to identify the unique selling points of curated songs. As a result, the ability to leverage personalized music recommendations to help users efficiently and accurately discover tracks of interest within the vast music library has become increasingly crucial.
### Motivation
This project focuses on predicting which songs will be recommended to users by utilizing a large dataset. Developing this recommendation system has the potential to transform how music apps interact with their users, enabling companies in the online media industry to optimize their return on investment. By analyzing the data collected from users, such as their music preferences and purchase behavior, businesses can better tailor their offerings and maximize revenue.



### Related Literature
Music recommendation systems have become crucial in enhancing user experience on platforms like Spotify and Pandora. With the advancement of machine learning techniques, several approaches have emerged to optimize the performance of these systems. This section reviews recent literature on collaborative filtering, content-based filtering, and hybrid models and discusses their application in industry settings such as Spotify.

Collaborative Filtering: Collaborative filtering (CF) predicts user preferences by analyzing historical behavior. Deldjoo et al. (2021) provided an extensive review of content-based music recommendation techniques, emphasizing the importance of CF in music recommendations. However, CF faces challenges like the cold-start problem and sparsity in user-item interactions. Jing et al. (2024) proposed a deep Bayesian network model that integrates heterogeneous data to address these issues and improve prediction accuracy.

Content-Based Filtering: Content-based filtering (CBF) recommends items by analyzing their attributes, such as genre, tempo, and lyrics. Luo et al. (2024) developed a multi-layered weighted hypergraph embedding learning method for diversified music recommendation, showcasing the effectiveness of CBF in capturing musical features and user preferences.

Hybrid Models: Hybrid recommendation systems combine the strengths of CF and CBF to improve recommendation accuracy. Deldjoo et al. (2021) emphasized the application of hybrid models in music recommendation, noting their advantage in handling complex user preferences and improving recommendation quality. 

### Industry Applications

Spotify: Spotify employs a hybrid recommendation system that integrates CF, CBF, and natural language processing to deliver personalized music recommendations. Gomez-Uribe and Hunt (2015) detailed Spotify’s approach, highlighting its success in maintaining high user engagement and satisfaction through personalized recommendations.

Pandora: Pandora uses the Music Genome Project, a content-based approach that analyzes songs based on hundreds of musical attributes. Furnas (2014) discussed Pandora’s strategy, emphasizing the platform’s ability to match songs with user preferences based on detailed song profiling.

### Relevant existing module
The following Python libraries and modules are highly suitable based on the project's requirements to develop a music recommendation system for Spotify, which involves collaborative filtering, content-based filtering, clustering, and time series analysis. These libraries are practical for handling the various aspects of the recommendation system and can help streamline the implementation.

### Collaborative Filtering

#### a. Surprise
Functionality: Surprise is a Python library designed to build and analyze recommender systems. It includes built-in collaborative filtering algorithms such as K-Nearest Neighbors (KNN), Singular Value Decomposition (SVD), and others.
Key Features: Supports KNN-based collaborative filtering.Includes matrix factorization techniques like SVD and SVD++. Built-in evaluation tools like cross-validation and error metrics (e.g., RMSE, MAE).
Rationale: Surprise is a powerful tool for collaborative filtering and matrix factorization, ideal for predicting user preferences based on past behavior. It is particularly useful for implementing collaborative filtering models efficiently.

#### b. Implicit
Functionality: Implicit is optimized for collaborative filtering using implicit feedback, such as user interactions with songs (e.g., plays or skips). It supports matrix factorization algorithms like Alternating Least Squares (ALS).
Key Features: Efficient for large, sparse datasets. Works well with implicit feedback data.
Provides implementations of ALS and other matrix factorization techniques.
Rationale: Implicit is highly suitable for scenarios where user feedback is implicit, like song plays, making it an excellent choice for music recommendation systems that deal with such data.

#### c. SciPy
Functionality: SciPy is a general-purpose library for scientific computing in Python, which includes a K-Nearest Neighbors (KNN) implementation useful for collaborative filtering.
Key Features: Implements KNN algorithms for user-based or item-based collaborative filtering. Supports sparse matrix operations, which is important for handling large, sparse datasets.
Rationale: SciPy is versatile for building collaborative filtering systems using KNN. It works well with smaller datasets or simpler collaborative filtering methods and can handle sparse matrices efficiently.

#### d. LightFM
Functionality: LightFM is a hybrid recommendation library that combines collaborative and content-based filtering. It is optimized for large-scale, sparse datasets.
Key Features: Supports hybrid collaborative filtering using implicit and explicit feedback.
Includes algorithms like Bayesian Personalized Ranking (BPR) and Weighted Approximate-Rank Pairwise (WARP). Efficient at handling large data and integrating song metadata.
Rationale: LightFM is an ideal choice for building a hybrid recommendation system that combines user behavior and song metadata. It is suitable for handling large-scale music recommendation tasks.

#### e. TensorFlow Recommenders (TFRS)
Functionality: TensorFlow Recommenders is a deep learning-based framework for building collaborative filtering models. It integrates well with TensorFlow and allows for building more complex models.
Key Features: Deep learning-based collaborative filtering models. Integrates easily with TensorFlow for more advanced feature representations. Supports both user-based and item-based models.
Rationale: TFRS is suitable for building deep learning-based recommendation systems, providing advanced techniques that can enhance recommendation accuracy by leveraging neural networks.

### Content-Based Filtering

a. spaCy
Functionality: spaCy is a powerful natural language processing (NLP) library that can be used to analyze song lyrics and metadata.
Key Features:Fast and efficient NLP processing.Pre-trained models for text processing tasks such as tokenization and named entity recognition.Integrates well with other machine learning libraries.
Rationale: spaCy is well-suited for text-based analysis, such as analyzing song lyrics to extract meaningful information for content-based filtering. It can be used to process metadata or lyrics for content-based recommendations.

b. Gensim
Functionality: Gensim is an NLP library used for topic modeling and document similarity. It can be used to analyze the semantic meaning of song lyrics or metadata.
Key Features:Efficient handling of large text corpora.Provides Word2Vec and other models for semantic analysis.Suitable for content-based recommendation using song descriptions or lyrics.
Rationale: Gensim is excellent for generating vector representations of song lyrics or descriptions, which can then be used to find similarities between songs and make recommendations based on content.

c. TfidfVectorizer (from sci-kit learn)
Functionality: TfidfVectorizer is a text feature extraction tool that converts text data into a numerical format using the Term Frequency-Inverse Document Frequency (TF-IDF) method.
Key Features: Converts song metadata or lyrics into a feature vector. Measures the importance of each word within the context of all documents (or songs).
Rationale: TfidfVectorizer is useful for transforming text data, like song lyrics, into features that can be used for content-based recommendations. It helps identify key terms that define a song’s content.

### K-Means Clustering

a. scikit-learn
Functionality: scikit-learn is a widely-used library for machine learning that includes K-Means clustering and other clustering algorithms.
Key Features: Implements K-Means and other clustering algorithms. Provides evaluation metrics like silhouette score for assessing clustering quality.
Rationale: scikit-learn is an excellent tool for clustering songs or users based on features like genre or listening history. It is easy to use and integrates well with other machine learning tasks.

b. HDBSCAN
Functionality: HDBSCAN is a density-based clustering algorithm that can identify clusters of arbitrary shapes and densities, making it useful for complex music data.
Key Features: Can identify arbitrarily shaped clusters. Works well with high-dimensional data.Does not require the number of clusters to be specified ahead of time.
Rationale: HDBSCAN is useful when K-Means is not suitable, especially for music data with varying density and complex relationships. It allows for better clustering in such cases.

c. MiniBatchKMeans (from scikit-learn)
Functionality: MiniBatchKMeans is an optimized version of K-Means that processes smaller batches of data, making it suitable for large datasets.
Key Features: Faster than traditional K-Means for large datasets.Suitable for clustering high-dimensional music feature data.
Rationale: MiniBatchKMeans is ideal for efficiently clustering large music datasets, making it well-suited for music recommendation systems that need to process a large number of songs.
1.3.4 Time Series Analysis

a. pandas
Functionality: pandas is a powerful data manipulation library that supports time series operations like rolling averages and time-based aggregations.
Key Features: Supports date-time functionality and time series operations. Can handle and manipulate time-based data, like user listening behavior over time.
Rationale: pandas is indispensable for handling time-based data and is ideal for analyzing trends in music preferences or user behavior over time.

b. stats models
Functionality: stats models provide tools for statistical modeling and include time series models like ARIMA for forecasting.
Key Features: Includes ARIMA and SARIMA models for time series forecasting. Can handle seasonal and trend components in time series data.
Rationale: stats models is well-suited for time series forecasting tasks and can be used to predict future trends in music preferences or user behavior.

c. Prophet
Functionality: Prophet is a forecasting tool developed by Facebook, designed to handle seasonal trends and missing data in time series.
Key Features: Handles daily, weekly, and yearly seasonalities. Built-in methods to handle missing data and outliers.
Rationale: Prophet is highly effective for forecasting music trends or user preferences, especially with seasonal patterns or incomplete data.
## Experimental Procedure 
### Data Preparation
Data cleaning is a crucial step in ensuring the accuracy and effectiveness of machine learning models. This process typically involves identifying and handling outliers (either by removal or imputation), addressing missing values, eliminating duplicate entries, and removing features with little or no significance. Additionally, data visualization plays a vital role in providing an initial understanding of the dataset, offering insights that guide preprocessing before model implementation. The group focused on removing irrelevant features and prioritizing highly correlated ones to maximize their influence on predicting satisfaction.
●	Data has been collected from the GitHub platform. (https://github.com/AmolMavuduru/SpotifyRecommenderSystem/tree/master/data) 
●	The data collected is over a span of 100 years, 1921 - 2020.
●	The datasets we are using for the project are - data.csv, data_by_year.csv, data_by_genre.csv.
●	Our main dataset data.csv(df_Spotify) consists of 170653 entries and 19 columns with non-null values. 
 ![image](https://github.com/user-attachments/assets/6ead219e-ff79-4dcd-af49-70c1352160a5)

Fig.1
●	data_by_year.csv shows audio features of songs in different years, it consists of 100 entries and 14 columns with non-null values. 
 ![image](https://github.com/user-attachments/assets/489dd9ec-9d02-4630-9c55-0dce1d9bc471)

Fig.2

●	data_by_genre.csv shows audio features for each genre, it consists of 2972 entries and 14 columns with non-null values. 

 ![image](https://github.com/user-attachments/assets/6095b713-e1aa-4aea-80b2-41691522e79f)

Fig.3

Clean and Transform Dataset: Data preparation is the method of cleaning and transforming raw data subsequent to processing and analysis. It's a crucial stage before processing that often entails reformatting data, making data corrections, and merging data sets to enrich data. To transform data into information and remove prejudice induced by poor data quality, it is vital to put it into context. We normalized our data by deleting duplicate and missing entries, removing superfluous symbols like quote marks and square brackets, and standardizing the dates
### Exploratory Data Analysis 
Exploratory Data Analysis (EDA) plays a crucial role in identifying relevant features such as song attributes, user preferences, and listening patterns. By analyzing data through visualizations and statistical methods, EDA helps uncover trends, correlations, and anomalies in user behavior and music features. Additionally, we can select features that are relevant to create a recommendation system.
●	The correlation heatmap shows the relationships between numerical variables in the Spotify dataset. It helps identify key features that are highly interrelated or independent, guiding the selection of relevant variables for building the recommendation system.
  ![image](https://github.com/user-attachments/assets/f91b4121-3627-4bd3-a557-edc6d1868b68)
Fig.4 Correlation Matrix between variables
●	Music over time: Using the data grouped by year, we can understand how the overall sound of music has changed from 1921 to 2020.

![image](https://github.com/user-attachments/assets/3cfdefbf-b1d9-4a5a-917f-645c745de482)

Fig.5 Number of songs per decade

 ![image](https://github.com/user-attachments/assets/e6b40539-3cc2-41a2-ae96-209392b3a5a6)

Fig.6 Popularity trends over years
 ![image](https://github.com/user-attachments/assets/9df9c6d2-3cff-4ccc-abb6-7ba9d67f883c)

Fig.7 Variable trends over years

●	Characteristics of different genres: This dataset contains the audio features for different songs along with the audio features for different genres. We can use this information to compare different genres and understand their unique differences in sound.
 ![image](https://github.com/user-attachments/assets/0d8e0bdf-8731-40bc-a67a-d6c21c6d9c9c)

Fig.8 Characteristics of different genres

 ![image](https://github.com/user-attachments/assets/189b54ee-0edc-49a2-8ec4-14aab53cbc4b)

Fig.9 Top genres by popularity
 ![image](https://github.com/user-attachments/assets/b7bbb505-0423-47bc-be54-c867dd0c42d0)

Fig.10 Danceability distribution for top 10 popular genres


 ![image](https://github.com/user-attachments/assets/71fca92b-d79c-4068-bd57-4055979d8af7)

Fig.11 Energy distribution for top 10 popular genres
 ![image](https://github.com/user-attachments/assets/da3aa920-cd2d-4639-a0ef-35a0d2d16eea)

Fig.12 Valence distribution for top 10 popular genres

 ![image](https://github.com/user-attachments/assets/b497dc86-2938-497c-ada7-a145abfbedab)

Fig.13 Acousticness distribution for top 10 popular genres
 ![image](https://github.com/user-attachments/assets/f5734d55-4fca-496a-a56b-9fb9a77a3f2b)

Fig.14 Instrumentalness distribution for top 10 popular genres


 ![image](https://github.com/user-attachments/assets/274e7802-524b-4245-963b-82e7dec6688f)

Fig.15 Genre WordCloud

## Techniques and Results 

### Time Series Analysis
Time series analysis is most effective for datasets that cover a long time span. Our database includes over a century of song recordings. As the term suggests, a time series represents data distributed over time. In this study, we use song features provided by Spotify to visualize trends over time and explore how our musical preferences have changed. The moving average method is applied in our model to uncover interesting patterns in the data.
Moving average smoothing is a simple yet effective technique for forecasting time series. It is useful for processing data, generating new features, and even making direct predictions. This method reduces small fluctuations in data between time periods by smoothing, aiming to minimize noise and reveal the underlying patterns or signals in the data. In time series analysis and forecasting, moving averages are one of the most commonly used methods for smoothing.
To calculate a moving average, a new data series is created by averaging the values of the original data over a specific period, known as the window size or window width. This window size determines how many data points are included in each average.
The "moving" in moving average indicates that the window slides across the time series, recalculating the average for each step to create the new series. The two most common types of moving averages are centered moving averages, which balance the window around a data point, and trailing moving averages, which calculate the average based on previous data points.
 ![image](https://github.com/user-attachments/assets/cde0be46-9259-482b-8cbd-ea0cd96d5160)

Fig.16
 ![image](https://github.com/user-attachments/assets/b696296f-87b9-4547-826c-1cadb7308512)

Fig.17
 ![image](https://github.com/user-attachments/assets/cfc54c14-fc96-4745-8416-309be901df71)

Fig.18
Popularity has steadily increased, with significant growth starting around the 1970s. Tempo has shown a gradual rise, peaking in the late 20th century before stabilizing in recent decades. Loudness has consistently increased over the years, reflecting modern production techniques that favor louder music. The use of moving averages effectively highlights these long-term trends by smoothing out short-term fluctuations, offering a clearer view of the overall evolution in music characteristics.
 ![image](https://github.com/user-attachments/assets/af44aed8-09ae-4ea7-ab48-0ae1132ecfe2)

Fig.19
We selected the audio features 'acousticness,' 'danceability,' 'energy,' 'liveness,' 'speechiness,' and 'valence' for various songs, as well as the corresponding features for different genres. This information allows us to compare genres and gain insights into their unique sonic characteristics.
 ![image](https://github.com/user-attachments/assets/e14238fd-0c37-4fa8-a924-2411b2cb2d75)

Fig.20
The analysis of normalized audio features over time reveals distinct trends in music evolution. Acousticness has declined, indicating a shift away from acoustic sounds, while danceability and energy have generally increased, reflecting a growing emphasis on upbeat and dynamic music. Liveness has remained stable, suggesting consistent inclusion of live performance elements, whereas speechiness shows low levels with occasional spikes, highlighting the rarity of spoken-word-heavy songs. Valence has slightly decreased in recent years, pointing to a potential shift towards less positive or more complex emotional themes in music. These trends illustrate the changing preferences and production styles in the music industry over the decades.
 ![image](https://github.com/user-attachments/assets/3766e423-865a-41ef-bec5-b687e2ea208c)

Fig.21
The seasonality analysis of music features reveals that attributes like danceability, duration, tempo, valence, and liveness have moderate seasonal patterns, suggesting some recurring trends over time. Meanwhile, features such as acousticness, energy, instrumentalness, loudness, and popularity show weak seasonality, indicating less consistent or less pronounced seasonal variations. Overall, some aspects of music demonstrate noticeable seasonal effects, while others vary with less regularity.
  ![image](https://github.com/user-attachments/assets/fd62d6cc-1f00-409b-a93a-06000e8885cc)
 ![image](https://github.com/user-attachments/assets/eb60c824-f786-47f0-b107-4df9fee768ae)
![image](https://github.com/user-attachments/assets/ff7c4f08-bafe-471a-b45d-9b8f451203e1)

fig.22
Overall, Exponential Smoothing performs best in forecasting trends for different music features, closely aligning with both long-term trends and short-term fluctuations in the actual data. Prophet captures the general trends but lacks precision in capturing short-term variations, while ARIMA shows significant divergence in most cases, particularly in recent predictions. Therefore, Exponential Smoothing is the most suitable tool for forecasting these features, providing a more accurate representation of how music characteristics evolve over time.
 ![image](https://github.com/user-attachments/assets/fb9c5f80-1dde-409a-82e5-12c9e0675f95)

Fig.23
In summary, Exponential Smoothing generally outperforms ARIMA and Prophet across different evaluation metrics, including MSE, RMSE, and MAE, indicating higher prediction accuracy for most audio features. ARIMA performs moderately well compared to Prophet but is still less accurate than Exponential Smoothing. Prophet shows relatively high error values and poor performance, particularly with significantly negative R2 values. Despite the better performance of Exponential Smoothing, all models have negative R2 values, suggesting limitations in effectively capturing the variance in the data.

### K-Means Clustering
Collaborative filtering is one of the earliest and most widely used techniques in recommendation systems. It often employs the K-Nearest Neighbor (KNN) method, leveraging users' historical data to calculate the similarity between different users based on their music preferences. This approach identifies the "nearest neighbor" users of the target user to evaluate and predict their preferences for specific items, thereby estimating how much the user might like a particular product.
On the other hand, clustering is an unsupervised learning technique that aims to identify homogeneous subgroups within the data. It groups data points into clusters so that points within a cluster are as similar as possible based on a chosen similarity measure, such as Euclidean distance. For this project, we will use K-means clustering, a popular algorithm for partitioning data into 'k' predefined distinct clusters. Each data point belongs to only one cluster, with the algorithm minimizing the sum of squared distances between the data points and the cluster centroids. K-means ensures that intra-cluster similarity is maximized while keeping clusters distinct from each other. Notably, the variation within clusters is inversely related to the homogeneity of the data points.
Euclidean Metric is given by:
 ![image](https://github.com/user-attachments/assets/49b2c841-8d48-4a22-8284-8024a5f1b222)

To determine the optimal value of K in K-Nearest Neighbors (KNN), it is essential to identify the K nearest neighbors of the unseen test data. The process involves assigning a class label to the test data point based on the majority vote or the highest probability derived from the class distribution of the K nearest data points.
When a test input xx is evaluated, it is classified into the class that has the highest frequency among its K neighbors. Essentially, the choice of K affects the model's performance:
●	Small K: The model becomes sensitive to noise and overfits the training data.
●	Large K: The model may oversimplify and fail to capture local patterns.
The class with the highest probability (or count) among the nearest neighbors is ultimately assigned to the test data point, making the selection of K crucial for achieving balanced accuracy and generalization.
 ![image](https://github.com/user-attachments/assets/3c7445ff-83c4-4c90-90d2-e3bb5acbefa1)

The genres have been grouped into clusters, and to enhance our understanding, we can visualize these clusters in a two-dimensional space. This allows us to see how different genres or songs relate to each other spatially.
To further this analysis, K-means clustering can be applied to the songs dataset. This approach enables us to group songs into distinct clusters based on their features, helping to identify patterns and similarities within the data. For instance, songs with similar characteristics, such as tempo, genre, or key, will be located closer to each other in the plot, forming tight clusters.
By exploring the resulting plot, we can observe that songs within the same cluster often share some degree of similarity. This is a fundamental concept in content-based recommendation systems, where the proximity or similarity of songs is used to suggest new tracks to users based on their preferences. Such visualizations and clustering analyses provide valuable insights for improving recommendation systems.

 ![image](https://github.com/user-attachments/assets/ce6410f5-76a9-4041-9dab-5427878b2cbe)

Fig.24

![image](https://github.com/user-attachments/assets/11518b95-af47-4343-8f30-763196e51ac8)
 
Fig.25
The optimal K value has been decided based on the SSE measure. The finalization of K=10 has been made based on the graphical representation.
 ![image](https://github.com/user-attachments/assets/697d9788-a8a0-4ac9-ae6d-441a5e40df20)

Fig.26
We further used Pyspark to perform the k mean clustering. We created a Spark Session using the following code: 
 ![image](https://github.com/user-attachments/assets/875cad88-bbd0-463a-b850-d2e76bf79caa)

We removed the columns which are not relevant and converted all the datatypes in string: 
 ![image](https://github.com/user-attachments/assets/fecc9baa-95a6-408a-9d8d-ea15829bf5ad)

Since all the variables we are examining are either numerical or discrete numeric, we will utilize a Vector Assembler to convert them into features. A Vector Assembler is a transformer that combines a collection of features into a single vector column, commonly referred to as an array of features. We will also scale these features accordingly.
 
![image](https://github.com/user-attachments/assets/faa58cec-e2a0-4a04-b948-a92673cd5a0f)
![image](https://github.com/user-attachments/assets/fd243658-646b-4604-a0dc-eb93b76da03c)
Next, we dove into the exciting world of K-means clustering.
 ![image](https://github.com/user-attachments/assets/88b68b9b-85ae-4bfb-b052-3ef6eaa0a916)

The error metric used is the Silhouette score: 
 ![image](https://github.com/user-attachments/assets/7c4e7fd7-82b7-4972-b30c-a609aae916e6)
Fig.27

Based on the above graph we decided to use the k value of 5, as there is a drastic decrease after that point.
### Recommender System 
Recommendation systems are an evolution of information filtering technology designed to provide personalized suggestions based on user profiles without requiring explicit user feedback. These systems rely on attribute-based methods; users and items have profiles detailing their attributes. If the attributes of a user profile align with those of an item, the system suggests that item to the user.
In our dataset, each song or artist is classified with over ten attributes. When a user selects a song, the system recommends other songs with similar genres. Developing a supervised model for an attribute-based recommendation system involves estimating users' likelihood of enjoying an item they have not interacted with. The target variable for the model is whether the user likes the item, and the input features are derived from user and item attributes. Attribute-based methods are straightforward to implement and adapt to new users or items.
Using K-means clustering, users are grouped into distinct categories. The resulting matrix displays top recommendations made by the system, prioritizing items with high ratings and play counts. Attributes such as ratings and play counts are key to generating recommendation track IDs. The outputs presented are generated using PySpark and sk-learn, demonstrating the system's functionality.
 ![image](https://github.com/user-attachments/assets/6bd29a4e-5396-4ded-9794-35d78ec4e2aa)

Fig.28
Fig 16 contains the actual list of genres the user listens to. Our recommended genres were 9,8 and 17. They are present in the below array. Therefore, our recommender system has made accurate recommendations. 
 ![image](https://github.com/user-attachments/assets/56002e2a-6344-4bd1-b5f6-4c2cb58ded4e)

Fig.29
## Conclusion
Spotify maintains detailed metadata and audio features for songs, which can be leveraged to develop robust music recommendation systems.
Improvement Idea: Based on the recommendation analysis, we can further enhance user engagement by curating a biweekly mix. This mix could feature playlists with a diverse selection of genres and tracks that align with the user's listening habits, offering a personalized and dynamic music experience.






 

## References 

1. Deldjoo, Y., Schedl, M., & Knees, P. (2021). Content-driven Music Recommendation: Evolution, State of the Art, and Challenges. *arXiv preprint arXiv:2107.11803*. ([arxiv.org](https://arxiv.org/abs/2107.11803))
2. Jing, E., Liu, Y., Chai, Y., Yu, S., Liu, L., Jiang, Y., & Wang, Y. (2024). Personalized Music Recommendation with a Heterogeneity-aware Deep Bayesian Network. *arXiv preprint arXiv:2406.14090*. ([arxiv.org](https://arxiv.org/abs/2406.14090))
3. Luo, C., Wen, L., Qin, Y., Yang, L., Hu, Z., & Yu, P. S. (2024). Against Filter Bubbles: Diversified Music Recommendation via Weighted Hypergraph Embedding Learning. *arXiv preprint arXiv:2402.16299*. ([arxiv.org](https://arxiv.org/abs/2402.16299))
4. Singh, R., & Kanuparthi, P. (2023). Related Rhythms: Recommendation System To Discover Music You May Like. *arXiv preprint arXiv:2309.13544*. ([arxiv.org](https://arxiv.org/abs/2309.13544))
5. Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix Recommender System: Algorithms, Business Value, and Innovation. *ACM Transactions on Management Information Systems (TMIS)*, 6(4), 1-19.
6. Furnas, G. W. (2014). The Pandora Music Genome Project. *Proceedings of the 2014 International Conference on Music Information Retrieval*, 1-6.

