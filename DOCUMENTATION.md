
<br>

<h1 align="center"> Uncertainty Estimation. Documentation</h1>

<br>

Documentation for the library for quantifying the uncertainty of predicting machine learning models

<br>

## Contents

1. [First steps at the start (initialization and preparation)](#getting-started)
    1. [Using the library](#using-the-library)
    2. [Initializing a class object](#initializing-a-class-object)
3. [Description of methods and parameters](#-description-of-methods-and-parameters-)
    1. [UncertaintyEstimation.calculate_logits_val](#uncertaintyestimationcalculate_logits_val)
    2. [UncertaintyEstimation.calculate_logits_mcd](#uncertaintyestimationcalculate_logits_mcd)
    3. [UncertaintyEstimation.calculate_classic_ue](#uncertaintyestimationcalculate_classic_ue)
    4. [UncertaintyEstimation.calculate_mcd_ue](#uncertaintyestimationcalculate_mcd_ue)
4. [More details](#more-details)
    1. [Entropy(ent)](#entropyent)
    2. [Maximum logits(max_logits)](#maximum-logitsmax_logits)
    3. [Maximum probability(max_prob)](#maximum-probabilitymax_prob)
    4. [Entropy by averaged softmaxs(mean_ent)](#entropy-by-averaged-softmaxsmean_ent)
    5. [Bald score(bald)](#bald-scorebald)
    6. [Variation ratio(vr)](#variation-ratiovr)
    7. [Probability variance(pv)](#probability-variancepv)
    8. [Maximum probability by averaged probabilities /softmax ratio(sr)](#maximum-probability-by-averaged-probabilities-softmax-ratiosr)
    9. [Maximum logits by averaged logits / logits ratio(lr)](#maximum-logits-by-averaged-logits--logits-ratiolr)

<br><br>

## First steps at the start (initialization and preparation) <a name="getting-started"></a> 

#### Using the library

&emsp;&emsp;
When using the library, you need to add it to the list of directories in which the Python interpreter searches for modules (sys.path). <br> An example of how to do this:

```python

import sys
import os

base_path = '/home/my_project/uncertainty_estimation_lib'
sys.path.append(os.path.join(base_path, 'src'))

```

#### Initializing a class object

&emsp;&emsp;
When initializing a class object, it is necessary to pass the config or the path to the config to it. An example of the config lies in */uncertainty_estimation_lib/configs/local_config.yaml* ([learn more about the config here](configs/README.md#Recommendations-for-config)) <br> For example:

```python
 
from uncertaintyestimation import UncertaintyEstimation

# you can create a class object in one of two ways
ue_obj_0 = UncertaintyEstimation(config)                                   
ue_obj_1 = UncertaintyEstimation('/home/my_project/configs/config.yaml')

```

Examples of the launch can be found in [LAUNCH_EXAMPLES.md](LAUNCH_EXAMPLES.md)

<br><br>

---

<h2 align="center"> Description of methods and parameters </h2>

---
<br>

---
## UncertaintyEstimation.calculate_logits_val 
--- 

* UncertaintyEstimation.**calculate_logits_val**(model, dataloader, mixed_precision=False) &emsp; [(usage example)](LAUNCH_EXAMPLES.md#using-the-uncertaintyestimationcalculate_logits_val-learn-more-about-the-method)



Calculate the logits. Logits are the values obtained at the output of the model when the validation samples pass through it. This function calculates the logits based on the validation sample. Next, these logits will be used to quantify the uncertainty of machine learning model predictions.

| Parameters: |     |
|------|------|
|      | **model: BertForSequenceClassification / another models**  <a name="information-about-model"></a>|  
|      | A ready-made trained model that has methods *predict*, *eval*, *train*. There is no need to worry about the model mode (train/eval), when calling the function, the model will switch to eval mode by default. The device on which calculations will be performed can be changed in the config in the *device* section (by default device='cpu:0'). <br><br>|  
|      | **dataloader: torch.utils.data.dataloader.DataLoader / an iterable object whose elements are a dictionary {name: data}** <a name="information-about-dataloader"></a>|
|      | Used to load data into the model. You can use an iterable object, each element of which is represented by a dictionary {name_0: data_0, name_1: data_1, ... }. If necessary, you can change the format of the data received in the data loader in the Uncertainty class in the __create_logits function. <br><br>|
|      | **mixed_precision: bool, default=False** | 
|      | Mixed precision is the use of 16—bit and 32-bit floating point types in the model during training so that it runs faster and uses less memory. If you want to save time and memory used, while it is not critical for you in some places to set the accuracy of float16 *mixed_precision=True*. |

| Returns: |     |
|------|------|
|      | **logits_array: ndarray of shape (n_samples, n_logits)** |
|      | A matrix of logits of the size *n_samples* (number of validation samples) by *n_logits* (number of logits/outputs of the model) <br><br>|

<br><br>

---
## UncertaintyEstimation.calculate_logits_mcd
---

* UncertaintyEstimation.**calculate_logits_mcd**(model, dataloader, mixed_precision=False, n_masks=None) &emsp; [(usage example)](LAUNCH_EXAMPLES.md#using-the-uncertaintyestimationcalculate_logits_mcd-learn-more-about-the-method)

Calculate the logits to estimate the uncertainty of the Monte Carlo dropout. Logits are the values obtained at the output of the model when the validation samples pass through it. This function calculates logits based on a validation sample, but the output is N different sets of logits. This is achieved by turning on/off neurons in a neural network using the Monte Carlo dropout method. Next, these logits will be used to quantify the uncertainty of machine learning model predictions.

| Parameters: |     |
|------|------|
|      | **model: BertForSequenceClassification / another models** <a name="information-about-model-mcd"></a> |  
|      | A ready-made trained model that has the methods *predict*, *eval*, *train*. Dropout layers must be embedded in the architecture of the model, because when using the method, the method of enabling and disabling dropout layers using the Monte-Carlo Dropouts method is involved. There is no need to worry about the model mode (train/eval), when calling the function, the model will switch to train mode by default (to activate dropouts). The device on which calculations will be performed can be changed in the configuration in the *device* section (by default device='cpu:0'). The calculation takes a lot of time (with an increase in the number of masks N, the calculation time increases), so it is recommended to use calculations on a video card. <br><br>|  
|      | **dataloader: torch.utils.data.dataloader.DataLoader / an iterable object whose elements are a dictionary {name: data}** <a name="information-about-dataloader-mcd"></a>|
|      | Used to load data into the model. You can use an iterable object, each element of which is represented by a dictionary {name_0: data_0, name_1: data_1, ... }. If necessary, you can change the format of the data received in the data loader in the Uncertainty class in the __create_logits function. <br><br>|
|      | **mixed_precision: bool, default=False** | 
|      | Mixed precision is the use of 16—bit and 32-bit floating point types in the model during training so that it runs faster and uses less memory. If you want to save time and memory used, while it is not critical for you in some places to set the accuracy of float16 *mixed_precision=True*. It is recommended to enable mixed_precision to speed up calculations. <br><br>|
|      | **n_masks: int, default=None** | 
|      | The parameter is responsible for the number of masks for Monte Carlo dropouts (the number of sets of logits that will result from calculations). As the number of masks increases, the calculation time increases proportionally. |

| Returns: |     |
|-----|-----|
|     | **logits_array: ndarray of shape (n_masks, n_samples, n_logits)** |
|     | Logit tensor of size *n_masks* (number of Monte Carlo dropout masks/number of logit sets) by *n_samples* (number of validation samples) by *n_logits* (number of model logits/outputs) <br><br>|

<br><br>

---
## UncertaintyEstimation.calculate_classic_ue
---

Calculate the uncertainty estimate in the classical way (entropy and/or maximum logits and/or maximum probability). Give an estimate of uncertainty for each sample, calculate and give a roc auc score, create a graph with a dropout of uncertainty based on the transmitted metric.

* UncertaintyEstimation.**calculate_classic_ue**(methods_ue='all', topn=1, saving_plots=True, saving_roc_auc=True, draw_plots=False, logits=None, 
targets=None, path_saving_plots=None, metrics='accuracy', metrics_average='micro') &emsp; [(usage example)](LAUNCH_EXAMPLES.md#using-the-uncertaintyestimationcalculate_classic_ue-learn-more-about-the-method)


| Parameters: |     |
|------|------|
|      | **methods_ue: ['ent', 'max_prob', 'max_logits'] or 'all', default='all'** |  
|      | Methods for calculating uncertainty estimates. This parameter is responsible for which methods will be used to estimate the uncertainty. By default, all methods are used. You can choose one, several or all. <br><br>  ***'ent'***: estimate the uncertainty by entropy. The entropy is calculated for each sample. Entropy is taken as the basis for estimating uncertainty [(more detailed)](#entropyent) <br><br>  ***'max_prob'***: estimate the uncertainty based on the maximum probability that the sample belongs to a certain class. The maximum probability for each sample is calculated. It is taken as a basis for estimating uncertainty [(more detailed)](#maximum-probabilitymax_prob) <br><br> ***'max_logits'***: estimate the uncertainty based on the maximum logits (outputs of the model). The maximum logits for each sample are calculated. This is taken as a basis for estimating uncertainty [(more detailed)](#maximum-logitsmax_logits) <br><br> ***'all'***: evaluate uncertainties for all methods. Uncertainties are calculated for all methods. On the basis of all methods, an estimation of uncertainty is made <br><br> |
|     | **topn: int, default=1** |
|     | This parameter specifies the number of top classes taken in one sample. In this module, it is used only with the *accuracy* metric. By default, the top 1 class is taken. When using topn=n, the top n classes are predicted. If at least one class of the top n classes is contained in target, then it is concluded that the model correctly determined the forecast and assigned the sample to a certain class. <br><br> |  
|     | **saving_plots: bool, default=True** |
|     | Save the schedule or not. The path to saving must be set in the config or via the *path_saving_plots* parameter (the parameter will be in priority). <br><br>|
|     | **saving_roc_auc: bool, default=True** |
|     | Save the roc_auc characteristic or not. The path to saving must be set in the config. <br><br>|
|     | **draw_plots: bool, default=False** |
|     | Display graphs immediately after calculation or not. By default, the parameter is set to False in order not to clog the workspace. <br><br> |
|     | **logits: array of float of shape(n_samples, n_logits) or None, default=None** |
|     | Parameter for loading logits. When you enter *None*, the logits will be loaded from the path defined in the config. You can also pass the logits obtained as a result of *Uncertainty.calculate_logits_val* (it is recommended to do this) or the logits obtained in another way. Takes a matrix of size *n_samples*(number of samples) by *n_logits*(number of logits). <br><br> |
|     | **targets: array of int of shape(n_samples,) or None, default=None**|
|     | Parameter for loading targets (true classes). When you enter *None*, the targets will be loaded from the path defined in the config. You can also pass targets as an array of integers. Accepts an array of size *n_samples*(number of samples). <br><br>|
|     | **path_saving_plots: str, default=None** |
|     | The path for saving graphs. When passing the path, the graphs will be saved to the specified directory. If the path is specified as None, then the graph saving paths will be taken from the config. By default, the parameter is None. <br><br> |
|     | **metrics: {'accuracy', 'f1', 'recall', 'precision'}, default='accuracy'**|
|     | The parameter specifies which metric to use for plotting graphs with dropout based on uncertainty estimation. Only one metric can be passed. By default, the metric *'accuracy'* is selected. <br><br> ***'accuracy'***: the accuracy (percentage) of correct forecasts is calculated. If the entire set of predicted labels for the sample strictly coincides with the true set of labels, then the accuracy of the subset is 1.0, otherwise it is 0.0. Based on this metric, a graph is constructed where the number of discarded (worst uncertainty) examples is postponed along the X axis, and the proportion of correct forecasts along the Y axis. With this metric, you can use the parameter *topn* >= 1. The result will be given for the top n classes. <br><br>  ***'f1'***: calculate the F1 score, also known as the balanced F-score or F-measure. The F1 indicator can be interpreted as the harmonic mean between accuracy and recall, where the F1 indicator reaches its best value at 1.0, and the worst indicator at 0.0. Based on this metric, a graph is constructed where the number of discarded (worst in uncertainty) examples is postponed along the X axis, and the F1 metric on the remaining (not discarded) along the Y axis/best) examples. <br><br> ***'recall'***: the *recall* metric is calculated (the ratio of true positive results to the sum of true positive results and the number of false negative results). The best result is 1.0, the worst will be 0.0. Based on this metric, a graph is constructed where the number of discarded (by the worst uncertainty) examples is postponed along the X axis, and the *recall* metric (on the remaining examples) along the Y axis. <br><br> ***'precision'***: the *precision* metric is calculated (the ratio of true positive results to the sum of true positive results and the number of false positive results). The best value of the metric can be 1.0, the worst can be 0.0. Based on this metric, a graph is constructed where the number of discarded (by the worst uncertainty) examples is postponed along the X axis, and the metric *precision* (on the remaining examples) along the Y axis. <br><br> |
|     | **metrics_average: {'micro', 'macro', 'weighted', 'binary', 'samples'}, default='micro'** |
|     | This parameter is required for multiclass/multilabel targets. This determines the type of averaging performed on the data. The description and functionality of this parameter is similar to that of the sklearn library. This parameter is used only with the metrics *f1*, *recall*, *precision*. When using the *accuracy* metric, the parameter will be ignored. <br><br> ***'micro'***: Calculate metrics globally by counting the total true positives, false negatives and false positives. <br><br> ***'macro'***: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account. <br><br> ***'weighted'***: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall. Weighted recall is equal to accuracy. <br><br> ***'binary'***: Report results for only one class. This applies only if the targets ( y_{true,pred}) are binary. <br><br> ***'samples'***: Calculate the metrics for each instance and find their average value (makes sense only for classification with multiple labels). <br><br>|

| Returns: |     |
|-----|-----|
|     | **classic_ue_dict, roc_auc_dict: dict({methods_ue: ue_array}), dict({methods_ue: roc_auc})** |
|     | Two dictionaries. The first dictionary: the keys are the names of the methods that were used for the calculation (str), the values are float arrays with shape (n_samples,) with calculated uncertainty for each sample. The second dictionary: the keys are the names of the methods that were used for the calculation (str), the values are float roc_auc. <br><br>|

<br><br>

---
## UncertaintyEstimation.calculate_mcd_ue
---

Calculate the uncertainty estimate using Monte Carlo dropout (mean entropy, and/or logits ratio, and/or softmax ratio, and/or probability variation, and/or variation ratio, and/or bald score). Give an estimate of uncertainty for each sample, calculate and issue a roc auc score, create a graph with a dropout of uncertainty based on the transmitted metric.

* UncertaintyEstimation.**calculate_mcd_ue**(methods_ue='all', topn=1, saving_plots=True, saving_roc_auc=True, draw_plots=False, logits=None, 
targets=None, mcd_logits_list=None, path_saving_plots=None, metrics='accuracy', metrics_average='micro')

[(usage example)](LAUNCH_EXAMPLES.md#using-the-uncertaintyestimationcalculate_mcd_ue-learn-more-about-the-method)

| Parameters: |     |
|------|------|
|      | **methods_ue: ['lr', 'sr', 'pv', 'vr', 'bald', 'mean_ent'] or 'all', default='all'** |  
|      | Methods for calculating uncertainty estimates. This parameter is responsible for which methods will be used to estimate the uncertainty. By default, all methods are used. You can choose one, several or all. <br><br>  ***'lr'***: estimate the uncertainty based on the maximum logits (model output) averaged over the logits of different Monte-Carlo dropout masks (MCD). The logits obtained on each MCD masks are averaged over the masks, then the maximum logit is taken as the basis for estimating uncertainty [(more detailed)](#maximum-logits-by-averaged-logits--logits-ratiolr) <br><br>  ***'sr'***: estimate the uncertainty based on the maximum probability (softmax is taken) of the probability-averaged (softmax) different Monte-Carlo dropout masks (MCD). The probabilities obtained on each MCD masks are averaged over the masks, then the maximum probability is taken as the basis for estimating uncertainty [(more detailed)](#maximum-probability-by-averaged-probabilities-softmax-ratiosr) <br><br> ***'pv'***: estimate uncertainty based on probability variance (softmax) among various Monte-Carlo dropout masks (MCD). The variance of probabilities is taken as the basis for estimating uncertainty [(more detailed)](#probability-variancepv) <br><br> ***'vr'***: estimate the uncertainty by the coefficient of variation (variation ratio). The most frequently encountered class (mode) among all Monte-Carlo dropout(MCD) masks is taken. The frequency of its occurrence is considered (the ratio of the number of times encountered to the total number of MCD masks. The frequency of occurrence (variation ratio) is taken as the basis for estimating uncertainty [(more detailed)](#variation-ratiovr) <br><br> ***'bald'***: estimate the uncertainty by bald score. The probability entropy (softmax) is calculated for each Monte-Carlo dropout mask (MCD), averaged over MCD masks, the so-called "expected entropy" is obtained. The "predictive entropy" is also considered - probabilities are averaged over MCD masks, entropy is considered. The difference between "predictive entropy" and "expected entropy" is taken as the basis for estimating uncertainty [(more detailed)](#bald-scorebald) <br><br> ***'mean_ent'***: estimate the uncertainty of the averaged entropy. Probabilities (softmax) are calculated for each dropout mask using the Monte Carlo method (MCD), averaged over MCD masks, entropy is calculated. The entropy of averaged probabilities is taken as the basis for estimating uncertainty [(more detailed)](#entropy-by-averaged-softmaxsmean_ent) <br><br> ***'all'***: evaluate uncertainties for all methods. Uncertainties are calculated for all methods. On the basis of all methods, an estimation of uncertainty is made <br><br> |
|     | **topn: int, default=1** |
|     | This parameter specifies the number of top classes taken in one sample. In this module, it is used only with the *accuracy* metric. By default, the top 1 class is taken. When using topn=n, the top n classes are predicted. If at least one class of the top n classes is contained in target, then it is concluded that the model correctly determined the forecast and assigned the sample to a certain class. <br><br> |  
|     | **saving_plots: bool, default=True** |
|     | Save the schedule or not. The path to saving must be set in the config or via the *path_saving_plots* parameter (the parameter will be in priority). <br><br>|
|     | **saving_roc_auc: bool, default=True** |
|     | Save the roc_auc characteristic or not. The path to saving must be set in the config. <br><br>|
|     | **draw_plots: bool, default=False** |
|     | Display graphs immediately after calculation or not. By default, the parameter is set to False in order not to clog the workspace. <br><br> |
|     | **logits: array of float of shape(n_samples, n_logits) or None, default=None** |
|     | Parameter for loading logits. When you enter *None*, the logits will be loaded from the path defined in the config. You can also pass the logits obtained as a result of *Uncertainty.calculate_logits_mcd* (it is recommended to do this) or the logits obtained in another way. Takes a matrix of size *n_samples*(number of samples) by *n_logits*(number of logits). <br><br> |
|     | **targets: array of int of shape(n_samples,) or None, default=None**|
|     | Parameter for loading targets (true classes). When you enter *None*, the targets will be loaded from the path defined in the config. You can also pass targets as an array of integers. Accepts an array of size *n_samples*(number of samples). <br><br>|
|     | **mcd_logits_list: array of float of shape(n_binary_masks, n_samples, n_logits) or None, default=None** |
|     | Parameter for loading Monte Carlo dropout (MCD) logits. If you enter *None*, the logits will be loaded along the path defined in the configuration. You can also pass the logits obtained as a result of *Uncertainty.calculate_logits_mcd* (it is recommended to do this) or the logits obtained in another way. Accepts a matrix of size *n_binary_masks*(number of MCD masks / sets of MCD logits) by *n_samples*(number of samples) by *n_logits*(number of logits in a sample). <br><br> |
|     | **path_saving_plots: str, default=None** |
|     | The path for saving graphs. When passing the path, the graphs will be saved to the specified directory. If the path is specified as None, then the graph saving paths will be taken from the config. By default, the parameter is None. <br><br> |
|     | **metrics: {'accuracy', 'f1', 'recall', 'precision'}, default='accuracy'**|
|     | The parameter specifies which metric to use for plotting graphs with dropout based on uncertainty estimation. Only one metric can be passed. By default, the metric *'accuracy'* is selected. <br><br> ***'accuracy'***: the accuracy (percentage) of correct forecasts is calculated. If the entire set of predicted labels for the sample strictly coincides with the true set of labels, then the accuracy of the subset is 1.0, otherwise it is 0.0. Based on this metric, a graph is constructed where the number of discarded (worst uncertainty) examples is postponed along the X axis, and the proportion of correct forecasts along the Y axis. With this metric, you can use the parameter *topn* >= 1. The result will be given for the top n classes. <br><br>  ***'f1'***: calculate the F1 score, also known as the balanced F-score or F-measure. The F1 indicator can be interpreted as the harmonic mean between accuracy and recall, where the F1 indicator reaches its best value at 1.0, and the worst indicator at 0.0. Based on this metric, a graph is constructed where the number of discarded (worst in uncertainty) examples is postponed along the X axis, and the F1 metric on the remaining (not discarded) along the Y axis/best) examples. <br><br> ***'recall'***: the *recall* metric is calculated (the ratio of true positive results to the sum of true positive results and the number of false negative results). The best result is 1.0, the worst will be 0.0. Based on this metric, a graph is constructed where the number of discarded (by the worst uncertainty) examples is postponed along the X axis, and the *recall* metric (on the remaining examples) along the Y axis. <br><br> ***'precision'***: the *precision* metric is calculated (the ratio of true positive results to the sum of true positive results and the number of false positive results). The best value of the metric can be 1.0, the worst can be 0.0. Based on this metric, a graph is constructed where the number of discarded (by the worst uncertainty) examples is postponed along the X axis, and the metric *precision* (on the remaining examples) along the Y axis. <br><br> |
|     | **metrics_average: {'micro', 'macro', 'weighted', 'binary', 'samples'}, default='micro'** |
|     | This parameter is required for multiclass/multilabel targets. This determines the type of averaging performed on the data. The description and functionality of this parameter is similar to that of the sklearn library. This parameter is used only with the metrics *f1*, *recall*, *precision*. When using the *accuracy* metric, the parameter will be ignored. <br><br> ***'micro'***: Calculate metrics globally by counting the total true positives, false negatives and false positives. <br><br> ***'macro'***: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account. <br><br> ***'weighted'***: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall. Weighted recall is equal to accuracy. <br><br> ***'binary'***: Report results for only one class. This applies only if the targets ( y_{true,pred}) are binary. <br><br> ***'samples'***: Calculate the metrics for each instance and find their average value (makes sense only for classification with multiple labels). <br><br>|

| Returns: |     |
|-----|-----|
|     | **mcd_ue_dict, roc_auc_dict: dict({methods_ue: ue_array}), dict({methods_ue: roc_auc})** |
|     | Two dictionaries. The first dictionary: the keys (str) are the names of the methods that were used for the calculation, the values are float arrays with shape (n_samples,) with calculated uncertainty for each sample. The second dictionary: the keys (str) are the names of the methods that were used for the calculation , the values are float roc_auc. <br><br>|

<br>

## More details

#### Entropy(ent)
---

What entropy means is best shown by a mathematical formula. Here the probabilities are taken after applying the softmax function, which makes a number from 0 to 1 from the logits (outputs of the model) (probability of the class)

$$entropy(X) = -\sum_{i=1}^k p_i * \log p_i $$

&emsp;&emsp;
where X is all states/probabilities, $p_i$ is the probability of the i-th class, k is number of classes 

$$p_i = softmax(z)_i = \frac{e^z_i}{^k\sum_{i=1} e^z_k} $$

&emsp;&emsp;
where z is a set of logits, k is the number of classes, i is the current class

<br>

#### Maximum logits(max_logits)
---

<br>

In one example, the number of logits is equal to the number of probable classes. We take the class corresponding to the maximum logit from the sample. This will be max_logit

$$ max\\_logits = max(logits) $$

<br>

#### Maximum probability(max_prob)
---

<br>

In one sample, the number of logits is equal to the number of probable classes. We apply the softmax function to the logits, getting the probabilities of the classes at the output. We take the class corresponding to the maximum probability from the sample. This will be max_prob

$$max\\_prob = max(softmax(logits))$$

[[what is softmax]](#entropyent)

<br>

#### Entropy by averaged softmaxs(mean_ent)
---

<br>

The average entropy is calculated according to the formulas given below. First, one sample is taken and probabilities (softmax) are calculated for each MCD mask, averaged over masks. Then, according to the averaged probabilities of belonging to each of the classes, entropy is taken. This will be the average entropy estimate using the MCD method

$$entropy = -\sum_{i=1}^k M_i * \log M_i $$

where Mi is the i-th averaged probability (over all Monte-Carlo dropout masks), k is number of classes

$$M_i = mean(p_i1, p_i2, p_i3 \dots p_iN)$$

where N is the number of MCD masks, and $p_i1, p_i2\dots p_iN$ are the i-th probabilities of belonging to a class in each mask

<br>

#### Bald score(bald)
---

Bald score - score, which is defined as the difference between predictive entropy and expected. The expected entropy is obtained as follows: the probability entropy (softmax) is calculated for each dropout mask using the Monte Carlo method (MCD), averaged over MCD masks. "Predictive entropy" is also taken into account - probabilities are averaged over MCD masks, entropy is taken into account. The calculation formulas are given below.

$$ bald\\_score = predictive\\_entropy - expected\\_entropy$$

$$ predictive\\_entropy = entropy(mean(softmax(logits))$$

$$ expected\\_entropy = mean(entropy(softmax(logits))) $$

[[what is softmax]](#entropyent) &emsp;&emsp; [[what is entropy]](#entropyent)


<br>

#### Variation ratio(vr)
---

To calculate the coefficient of variation (variation ratio), it is necessary to predict the class for each sample in each MCD mask (in this case, this is done as the [maximum probability](#maximum-probabilitymax_prob) of belonging to the class, i.e. the maximum from [softmax](#entropyent) from the logits in the sample). Then the statistical mode of predicted classes is taken among all the MCD masks of a given sample and the number of such predictions is calculated (i.e. we find the most frequently encountered class, we count how many times it has met). Then we divide the number of forecasts that are equal to the statistical mode by the total number of forecasts (i.e. by the number of MCD masks). We subtract the result from one - we found the coefficient of variation. Its value will be from 0 to 1. Thus, the greater the coefficient of variation, the fewer forecasts are equal to the statistical mode, which means there will be more uncertainty.

$$ variation\\_ratio = 1 - \frac{k}{N} $$

where N is the total number of forecasts (which is equal to the number of MCD masks), and k is the number of forecasts that are equal to the statistical mode of all forecasts of a given sample for each MCD mask

<br>

#### Probability variance(pv)
---

Probability variation (as already seen by the name) is calculated by the variance formula, i.e. probabilities (softmax) are calculated for each MCD mask of a given sample, averaged over MCD masks. The sum of the square of the difference
between the averaged probabilities and the current probabilities is taken, divided by the number of such squares of the difference (the number of masks). Also, one is subtracted from the denominator - this is due to degrees of freedom). Below is the formula for counting.

$$ probability\\_variance = \frac{^n\sum_{i=1} (x_i - \overline x)^2}{n-1} $$

where n is the number of MCD masks, $x_i$ is the probabilities (softmax) of the sample of the i-th MCD mask, $\overline x$ is the averaged probabilities (softmax) for MCD masks

[[what is softmax]](#entropyent)

<br>

#### Maximum probability by averaged probabilities /softmax ratio(sr)
---

Maximum probability by averaged probabilities /softmax ratio - an estimate that is obtained as a result of averaging over the masks of MCD probabilities (softmax), and then finding the maximum among the averaged probabilities. The formulas are given below.

$$ softmax\\_ratio = max(mean(softmax(logits))) $$

[[what is softmax]](#entropyent)

<br>

#### Maximum logits by averaged logits / logits ratio(lr)
---

Maximum logits by averaged logits / logits ratio - an estimate that is obtained by averaging logits by MCD masks in one sample, and then taking the maximum from the resulting logits. 

$$ logits\\_ratio = max(mean(logits)) $$

<br>

[[back/up]](#-uncertainty-estimation-documentation)
