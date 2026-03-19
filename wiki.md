# Wiki Activity - Clustering

Two clustering visuals were explored to understand how the K-Means algorithm works in practice. The animations demonstrated how clusters form dynamically through iterative reassignment of data points and recalculation of centroids.

### Relationship between the Animation and Algorithm Logic

The [first animation](http://shabal.in/visuals/kmeans/6.html) showed how clusters gradually emerge when points are grouped based on proximity, reflecting the logic of K-Means (Kavlakoglu & Winland, 2025):

The number of clusters (k) are chosen:
k centroids are initialised where k is equal to the number of clusters chosen for the dataset.

Centroids are assigned:
Using the expectation maximization machine learning algorithm, each datapoint is assigned to its closest centroid based on distance. In the animation, this was visualised by points moving withing their centroid. Using maximisation, the mean of the centroid is computed and this reassings the cluster center.

Iteration until convergence:
These steps repeat until the centroids stop moving and points no longer change clusters or until the maximum number of iterations has been reached. By this point there will be algorithm convergene, where the total distance between points and their cluster centres is minimised.

### Insights from the Second Animation (Uniform Points)
For the [second algorithm](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/), I chose the data points, using the 'Uniform Points' option which illustrated additional behaviours:

This time, the Voronoi regions were shown which clearly showed how cluster boundaries are determined. Using coloured points and Voronoi regions, it was easily illustrated how different points changed cluters with each iteration. When centroids were poorly initialised, clusters were uneven or unstable, highlighting that final clustering depends on starting positions.

The animation also showed that K-Means performs best when clusters are roughly spherical and evenly distributed, but struggles with irregular shapes.

### Reflection
Using animations, it was easier to understand how K-Means Clustering was applied, since this is not readily apparent from the equations themselves. The animations showed how clustering performance depends heavily on dataset characteristics such as scale, distribution, and shape. This reinforces that machine learning algorithms are not universally suitable and must be carefully matched to the dataset. Initialising the wrong centroid or the wrong number of clusters can lead to longer run time and badly assigned clusters. Additionally, clusters have to be similar in size and with no obvious outliers (may lead to overfitting) or density variations. This may lead to bias and misclassification, especially among smaller clusters (Kavlakoglu & Winland, 2025). As explained by Harris (2014) in the second animation, this issue could be solved by models such as the Gaussian Mixture node. 

In real-world applications, incorrect clustering could lead to biased decision-making or unfair treatment of individuals. Machine learning professionals therefore have a responsibility to understand the algorithm limitations, validate results, and communicate uncertainty transparently.


### References
Harris, N. (2014) *Visualizing K-means clustering, Naftali Harris*. Available at: https://www.naftaliharris.com/blog/visualizing-k-means-clustering/ (Accessed: 19 March 2026). 
Kavlakoglu, E. and Winland, V. (2025) *What is K-means clustering?*, IBM. Available at: https://www.ibm.com/think/topics/k-means-clustering (Accessed: 19 March 2026). 

