import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2
def compute_tp_fp_rates(saliency_map, ground_truth_map, thresholds):
    TPRs = []
    FPRs = []
    
    total_fixations = np.sum(ground_truth_map)  # Total number of fixated pixels (TP + FN)
    total_non_fixations = np.size(ground_truth_map) - total_fixations  # Total non-fixated pixels (FP + TN)
    
    for threshold in thresholds:
        # Apply threshold
        prediction = (saliency_map >= threshold).astype(int)
        
        TP = np.logical_and(prediction == 1, ground_truth_map == 1).sum()
        FP = np.logical_and(prediction == 1, ground_truth_map == 0).sum()
        FN = np.logical_and(prediction == 0, ground_truth_map == 1).sum()
        TN = np.logical_and(prediction == 0, ground_truth_map == 0).sum()
        
        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        
        TPRs.append(TPR)
        FPRs.append(FPR)
    
    return TPRs, FPRs

# Generating thresholds
saliency_map_path = r"/data/jiaoshengjie/Code/Saliency_transformer/results/Fairchild-CemeteryTree.jpg"
saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
ground_truth_map_path = r"/data/jiaoshengjie/Code/Saliency_transformer/IMLHDR/density/Fairchild-CemeteryTree.png"
ground_truth_map = cv2.imread(ground_truth_map_path, cv2.IMREAD_GRAYSCALE)

saliency_min, saliency_max = saliency_map.min(), saliency_map.max()
thresholds = np.linspace(saliency_min, saliency_max, num=50)  # Adjust num for finer granularity

# Compute rates
TPRs, FPRs = compute_tp_fp_rates(saliency_map, ground_truth_map, thresholds)

# Compute AUC
roc_auc = auc(FPRs, TPRs)
print(roc_auc)
# # Plotting the ROC curve
# plt.figure()
# plt.plot(FPRs, TPRs, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
