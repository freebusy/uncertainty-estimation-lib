# Recommendations for config
Here are recommendations for filling in the **config.yaml** file for correct operation. (The configuration view itself is presented below after the recommendations) <br>

* You need to fill in the following variables to specify the **paths** to save/upload files: **val_logits_path, mcd_logits_path, targets_path, plots_path, roc_auc_dict_path**.

* It is necessary to select the **device** on which the calculations will be carried out. By default, **device**='cpu:0' is selected, but it is recommended to use a video card to speed up calculations (e.g. **device**='cuda:0')

* Specify the number of masks (**n_binary_masks**) for Monte-Carlo dropout: by default **n_binary_masks**=5, it is recommended to use 30 or more

Other names can be used by default/filled in at your discretion

<br><br>
### basic type of config:

```yaml

paths: 
  saved_files_paths:
    val_logits_path: '/home/my_project/results/val_logits/'
    mcd_logits_path: '/home/my_project/results/mcd_logits/'
    targets_path: '/home/my_project/results/targets/'
    plots_path: '/home/my_project/results/plots/'
    roc_auc_dict_path: '/home/my_project/results/roc_auc_score/'
    names:
      logits_val_name: 'logits_val'
      targets_val_name: 'targets_val'
      plots_ue_classic_name: 'plot_classic_ue'
      plots_ue_mcd_name: 'plot_mcd_ue'
      roc_auc_dict_classic_name: 'roc_auc_score_classic_ue'
      roc_auc_dict_mcd_name: 'roc_auc_score_mcd_ue'
      logits_mcd_name: 'logits_mcd'
      logits_mcd_all_name: 'logits_mcd_all'

monte_carlo_dropout_settings:
  # this is not enough, it takes 30-50
  n_binary_masks: 5 

device: 'cpu:0'

```
