
<h1 align="center"> Launch examples </h1>

<br>

Here are examples of loading a library, defining a class object, and using various methods with different parameters

## Contents

1. [First steps at the start (initialization and preparation)](#first-steps-at-the-start-initialization-and-preparation)
    1. [Using the library](#using-the-library)
    2. [Initializing a class object](#initializing-a-class-object)
2. [Using the UncertaintyEstimation.calculate_logits_val](#using-the-uncertaintyestimationcalculate_logits_val-learn-more-about-the-method)
3. [Using the UncertaintyEstimation.calculate_logits_mcd](#using-the-uncertaintyestimationcalculate_logits_mcd-learn-more-about-the-method)
4. [Using the UncertaintyEstimation.calculate_classic_ue](#using-the-uncertaintyestimationcalculate_classic_ue-learn-more-about-the-method)
    1. [basic example(calculates the uncertainty of the model by entropy, maximum logits and maximum probability)](#basic-example-classic-ue)
    2. [calculate the uncertainty estimate based on the logits set via the config](#based-on-the-logits-by-config-classic-ue)
    3. [calculate the entropy (you can substitute one or more methods) uncertainty estimate](#the-entropy-classic-ue)
    4. [calculate using two methods (you can substitute one or more methods) uncertainty estimate](#using-two-methods-classic-ue)
    5. [calculate the uncertainty estimate based on the transmitted metric](#transmitted-metric-classic-ue)
    6. [calculate the uncertainty estimate for the top N classes](#topn-classic-ue)
    7. [calculate the uncertainty estimate for all methods and immediately output a graph](#with-graph-classic-ue)
5. [Using the UncertaintyEstimation.calculate_mcd_ue](#using-the-uncertaintyestimationcalculate_mcd_ue-learn-more-about-the-method)
    1. [basic example (calculates the uncertainty of the model by averaged entropy, maximum logits, maximum probability, probability variance, variation ratio, bald score)](#basic-example-mcd-ue)
    2. [calculate the uncertainty estimate based on the mcd_logits set via the config](#based-on-the-mcd-logits-by-config-mcd-ue)
    3. [calculate the mean entropy (you can substitute one or more methods) uncertainty estimate](#mean-entropy-mcd-ue)
    4. [calculate using some methods (you can substitute one or more methods) uncertainty estimate](#some-methods-mcd-ue)
    5. [calculate the uncertainty estimate based on the transmitted metric](#metric-mcd-ue)
    6. [calculate the uncertainty estimate for the top N classes](#topn-mcd-ue)
    7. [calculate the uncertainty estimate for all methods and immediately output a graph](#with-graph-mcd-ue)

## First steps at the start (initialization and preparation)

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
#### Using the UncertaintyEstimation.calculate_logits_val ([learn more about the method](DOCUMENTATION.md#uncertaintyestimationcalculate_logits_val))

&emsp;&emsp;
Using the UncertaintyEstimation.**calculate_logits_val** method to get logits (the model is defined as *[model](DOCUMENTATION.md#information-about-model)*, the dataloader as *[val_dataloader](DOCUMENTATION.md#information-about-dataloader)*):

```python
 
ue_obj = UncertaintyEstimation(config)  

# classic logits calculation
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)

# logits calculation with mixed precision
val_logits = ue_obj.calculate_logits_val(model, val_dataloader, mixed_precision=True)

 ```
 
 #### Using the UncertaintyEstimation.calculate_logits_mcd [(learn more about the method)](DOCUMENTATION.md#uncertaintyestimationcalculate_logits_mcd)
 
 &emsp;&emsp;
 Using the UncertaintyEstimation.**calculate_logits_mcd** method to get logits (the model is defined as *[model](DOCUMENTATION.md#information-about-model-mcd)*, the dataloader as *[val_dataloader](DOCUMENTATION.md#information-about-dataloader-mcd)*):

```python
 
ue_obj = UncertaintyEstimation(config)  

# Monte-Carlo dropouts logits calculation
mcd_logits = ue_obj.calculate_logits_mcd(model, val_dataloader)

# Monte-Carlo dropouts logits calculation with mixed precision
mcd_logits = ue_obj.calculate_logits_mcd(model, val_dataloader, mixed_precision=True)

 ```
 
 or if the number of Monte-Carlo dropouts masks (in this example, let's take 10 masks) is specified explicitly, and not in the config (priority over the config)
 
 ```python
 
ue_obj = UncertaintyEstimation(config)  

mcd_logits = ue_obj.calculate_logits_mcd(model, val_dataloader, n_masks=10)

 ```

#### Using the UncertaintyEstimation.calculate_classic_ue [(learn more about the method)](DOCUMENTATION.md#uncertaintyestimationcalculate_classic_ue)

&emsp;&emsp;
Using the UncertaintyEstimation.**calculate_classic_ue** method to calculate classical uncertainty by entropy, maximum logits, maximum probability (the model is defined as *[model](DOCUMENTATION.md#information-about-model)*, the dataloader as *[val_dataloader](DOCUMENTATION.md#information-about-dataloader)*). In all of the following examples, it is assumed that the target variables lie in the array *targets* <br> Validation logits are obtained using *[Uncertainty.calculate_logits_val](#using-the-uncertaintyestimationcalculate_logits_val-learn-more-about-the-method)*(Recommended)
* basic example (calculates the uncertainty of the model by entropy, maximum logits and maximum probability) <a name="basic-example-classic-ue"></a>
 
 ```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)
# or download ready-made val_logits yourself
# val_logits = numpy.load('validation_logits.npy', allow_pickle=True)

classic_ue_dict, roc_auc_dict = ue_obj.calculate_classic_ue(logits=val_logits, targets=targets)

```

* calculate the uncertainty estimate based on the logits set via the [config](configs/README.md#Recommendations-for-config) <a name="based-on-the-logits-by-config-classic-ue"></a>

 ```python

ue_obj = UncertaintyEstimation(config) 

classic_ue_dict, roc_auc_dict = ue_obj.calculate_classic_ue()

```

* calculate the entropy (you can substitute one or more methods) uncertainty estimate <a name="the-entropy-classic-ue"></a>

```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)

classic_ue_dict, roc_auc_dict = ue_obj.calculate_classic_ue(methods_ue=['ent'], logits=val_logits, targets=targets)

```

* calculate using two methods (you can substitute one or more methods) uncertainty estimate <a name="using-two-methods-classic-ue"></a>

```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)

classic_ue_dict, roc_auc_dict = ue_obj.calculate_classic_ue(methods_ue=['max_logits', 'max_prob'], 
                                                            logits=val_logits, targets=targets)

```

* calculate the uncertainty estimate based on the transmitted metric (default metrics='accuracy') <a name="transmitted-metric-classic-ue"></a>

```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)

classic_ue_dict, roc_auc_dict = ue_obj.calculate_classic_ue(metrics='f1', metrics_average='micro', 
                                                            logits=val_logits, targets=targets)

```

* calculate the uncertainty estimate for the top N classes <a name="topn-classic-ue"></a>

```python

N = 3
ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)

classic_ue_dict, roc_auc_dict = ue_obj.calculate_classic_ue(metrics='accuracy', topn=N, logits=val_logits, targets=targets)

```

* calculate the uncertainty estimate for all methods and immediately output a graph <a name="with-graph-classic-ue"></a>

```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)

classic_ue_dict, roc_auc_dict = ue_obj.calculate_classic_ue(draw_plots=True, logits=val_logits, targets=targets)

```

#### Using the UncertaintyEstimation.calculate_mcd_ue [(learn more about the method)](DOCUMENTATION.md#uncertaintyestimationcalculate_mcd_ue)

&emsp;&emsp;
Using Uncertainty.**calculate_mcd_ue** to calculate the uncertainty estimate of the model using the Monte Carlo dropouts (MCD) method for averaged entropy, maximum logits (averaging over a set of masks (MCD)), maximum probability (averaging over a set of masks MCD), probability variance, covariance coefficient, bald score. (the model is defined as *[model](DOCUMENTATION.md#information-about-model-mcd)*, the dataloader as *[val_dataloader](DOCUMENTATION.md#information-about-dataloader-mcd)*). In all the following examples, it is assumed that the target variables are in the array *targets* <br> (Recommended) Validation logits were obtained using *[Uncertainty.calculate_logits_val](#using-the-uncertaintyestimationcalculate_logits_val-learn-more-about-the-method)*, MCD logits were obtained using *[Uncertainty.calculate_logits_mcd](#using-the-uncertaintyestimationcalculate_logits_mcd-learn-more-about-the-method)*

 * <a name="basic-example-mcd-ue"></a> basic example (calculates the uncertainty of the model by averaged entropy, maximum logits (averaging over a set of masks (MCD)), maximum probability (averaging over a set of masks MCD), probability variance, covariance coefficient, bald score) 
 
 ```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)
mcd_logits = ue_obj.calculate_logits_mcd(model, val_dataloader)

# or download ready-made val_logits and mcd_logits yourself
# val_logits = numpy.load('validation_logits.npy', allow_pickle=True)
# mcd_logits = numpy.load('mcd_logits.npy', allow_pickle=True)

mcd_ue_dict, roc_auc_dict = ue_obj.calculate_mcd_ue(logits=val_logits, targets=targets, mcd_logits_list=mcd_logits)

```

* calculate the uncertainty estimate based on the mcd_logits set via the [config](configs/README.md#Recommendations-for-config) <a name="based-on-the-mcd-logits-by-config-mcd-ue"></a>

 ```python

ue_obj = UncertaintyEstimation(config) 

mcd_ue_dict, roc_auc_dict = ue_obj.calculate_mcd_ue()

```

* calculate the mean entropy (you can substitute one or more methods) uncertainty estimate <a name="mean-entropy-mcd-ue"></a>

 ```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)
mcd_logits = ue_obj.calculate_logits_mcd(model, val_dataloader)

mcd_ue_dict, roc_auc_dict = ue_obj.calculate_mcd_ue(methods_ue=['mean_ent'], logits=val_logits, targets=targets, 
                                                    mcd_logits_list=mcd_logits)

```

* calculate using some methods (you can substitute one or more methods) uncertainty estimate <a name="some-methods-mcd-ue"></a>

```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)
mcd_logits = ue_obj.calculate_logits_mcd(model, val_dataloader)

mcd_ue_dict, roc_auc_dict = ue_obj.calculate_mcd_ue(methods_ue=['mean_ent', 'bald', 'vr'], logits=val_logits, 
                                                    targets=targets, mcd_logits_list=mcd_logits)

```

* calculate the uncertainty estimate based on the transmitted metric (default metrics='accuracy') <a name="metric-mcd-ue"></a>

```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)
mcd_logits = ue_obj.calculate_logits_mcd(model, val_dataloader)

mcd_ue_dict, roc_auc_dict = ue_obj.calculate_mcd_ue(methods_ue='all', metrics='f1', metrics_average='micro', 
                                                    logits=val_logits, targets=targets, mcd_logits_list=mcd_logits)

```

* calculate the uncertainty estimate for the top N classes <a name="topn-mcd-ue"></a>

```python

N = 3
ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)
mcd_logits = ue_obj.calculate_logits_mcd(model, val_dataloader)

mcd_ue_dict, roc_auc_dict = ue_obj.calculate_mcd_ue(metrics='accuracy', topn=N, logits=val_logits, 
                                                    targets=targets, mcd_logits_list=mcd_logits)

```

* calculate the uncertainty estimate for all methods and immediately output a graph <a name="with-graph-mcd-ue"></a>

```python

ue_obj = UncertaintyEstimation(config) 
val_logits = ue_obj.calculate_logits_val(model, val_dataloader)
mcd_logits = ue_obj.calculate_logits_mcd(model, val_dataloader)

mcd_ue_dict, roc_auc_dict = ue_obj.calculate_classic_ue(draw_plots=True, logits=val_logits, 
                                                        targets=targets, mcd_logits_list=mcd_logits)

```



