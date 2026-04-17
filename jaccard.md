# Jaccard Coefficient Calculations

### Task
The following dataset was given, showing the pathological test results for three individuals:

| Name | Gender | Fever | Cough | Test-1 | Test-2 | Test-3 | Test-4 |
|------|--------|-------|-------|--------|--------|--------|--------|
| Jack | M      | Y     | N     | P      | N      | N      | A      |
| Mary | F      | Y     | N     | P      | A      | P      | N      |
| Jim  | M      | Y     | P     | P      | N      | N      | A      |

The Jaccard Coefficient was calculated for the following pairs:

- (Jack, Mary) <br>
- (Jack, Jim)<br>
- (Jim, Mary)<br>

The Jaccard Coefficient is calculated as follows (Jaccard Similarity, 2025):<br>
J(A,B) = ($\frac{|A∩B|}{|A∪B|}$) 

where |A∩B| shows the similarities between the sets and |A∪B| shows the number of unique elements in the set.

The output can range from 0 to 1, where:
- J(A,B) = 0 --> The sets do not have any common elements<br>
- J(A,B) = 1 -->  The sets are identical<br>

For 2 binary vectors:
J(A,B) = ($\frac{M11}{M01+M10+M11}$) 

where, <br>
M11 = Number of positions where both vectors have 1<br>
M10 = Positions where A has 1 and B has 0<br>
M01 = Positions where A has 0 and B has 1<br>

Treating Y or P as 1, and N or A as 0.

- (Jack, Mary) = ($\frac{2}{1+1+2}$) = 0.50<br>
- (Jack, Jim) = ($\frac{2}{1+0+2}$) = 0.67<br>
- (Jim, Mary) = ($\frac{2}{1+1+2}$) = 0.50<br>

Therefore this shows that the set most similar was (Jack, Jim) with a Jaccard coefficient of 0.67 and the others were equally less similar with the same Jaccard coefficient of 0.50.

### Reflection
The task required converting categorical pathological test outcomes into binary values which represent the presence or absence of symptoms or conditions to compute the similarity between pairs of individuals, demonstrating how similarity measures can be applied to real-world datasets. This exercise helped me learn about the challenges of real-world datasets which are often heterogeneous and contain missing values. Converting symbolic data into binary respresentations requires assumptions about what constitutes meaningful similarity. This highlights the importance of data preprocessing and domain knowledge when applying machine learning techniques. Furthermore, small datasets, such as the one used, can limit the reliability and generalisibilty of similarity measures.

This exercise also raised legal, social, ethical, and professional considerations. Medical data is highly sensitive, and machine learning professionals must ensure that similarity analysis does not lead to misclassification, discrimination, or breaches of confidentiality. Decisions based on similarity metrics could influence diagnosis and more which could affect the patient's health. Professionals therefore have a responsibility to apply data governance practices and ensure transparency in their modelling assumptions.


### References
*Jaccard similarity* (2025) *GeeksforGeeks*. Available at: https://www.geeksforgeeks.org/python/jaccard-similarity/ (Accessed: 19 March 2026). 

[← Back to Home](https://mmiz02.github.io/portfoliomachinelearning/)
