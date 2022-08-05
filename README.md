# KFCPFSKLD
<!DOCTYPE html>
<html>
	<head>
		<title>readme</title>
	</head>
	<body>
    <h1> A Noise Robust Kernel Fuzzy Clustering based on Picture Fuzzy Sets and KL Divergence Measure for MRI Image Segmentation </h1>
		<p>The user is supposed to clone the whole directory into their system. They are supposed to run the file named “Main.ipynb”. It is a demonstration of the segmentation of the brain MRI using the proposed methodology. Initially several modules, prebuilt as well as built by the authors would be imported and installed. On running the “Main.ipynb”, the user can find confusion matrices providing the performance of classification of various regions for various noise levels. Furthermore, the user can find a dataframe for various images with rician noises varying from 5% to 15% and for each one of these several metrics to judge the performance. Several performance metrics like dice score, jaccard score have been calculated for regions separately i.e. white matter, grey matter, CSF and background. The performance metrics are as follows : <br>
1) Jaccard Score : jac <br>
2) Dice Score : dic <br>
3) False Positive Ratio: FPR <br>
4) False Negative Ratio: FNR <br>
5) Partition Entropy: PE <br>
6) Partition Coefficient : PC <br>
7) Average Segmentation Accuracy : Acc 

Further in the directory, users can find another directory with the name “Test_Img” which further contains directories named RN and GN which contains images with Rician and Gaussian Noises respectively. The Test_Img also contains a file named “NewImg2.tiff” which is the ground truth. 

Users can also find several code files in the KFCPFKLD directory i.e. <br>
"Utils" : It contains all the tools and functionalities required by the code “Main.ipynb” file e.g. reshaping_linear, grounded_imaging, processing, fuzzy_to_crisp, clustering, indexing_final, append_result, tot_dice_score, tot_jaccard_score  <br>
"Fcm" : It contains code for the basic fuzzy c means clustering algorithm which is required by the code “Main.ipynb” file. <br>
"Kfcpfskld" : It contains code for the proposed methodology i.e. A Noise Robust Kernel Fuzzy Clustering based on Picture Fuzzy Sets and KL Divergence Measure for MRI Image Segmentation. <br>
</p>
	</body>
</html>
