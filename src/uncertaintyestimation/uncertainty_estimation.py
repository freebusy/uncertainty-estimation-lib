import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import copy
import yaml
from typing import Dict, List, Union


class UncertaintyEstimation():
    """
    The Uncertainty Estimation class helps to get an estimate of uncertainty in 
    natural language processing (NLP) tasks. Using this class, you can get a 
    quantitative estimate of uncertainty using the classical approach of entropy, 
    maximum logits, maximum probability, or using the Monte Carlo dropout method 
    (MCD) of averaged entropy, scoring, coefficient of variation, probability 
    variation, softmax coefficient, logits coefficient

    Atributes
    ---------
    conf : str or object
        configuration to run (the config itself or the path to it is passed)
    topn : int, default=1
        the number of top forecasts taken into account in the calculation
    saving_plots : bool, default=True
        save the schedule or not
    saving_roc_auc : bool, default=True
        save the roc_auc score or not
    draw_plots : bool, default=False
        drawing a graph at once
    path_saving_plots : str, default=None
        the way to save graphs
    metrics : str, default='accuracy'
        the metric by which the method works
    metrics_average : str, default='micro'
        the type of averaging performed for the data
    mixed_precision : bool, default=False
        mixed precision to include or not
    n_masks : int, default=None
        number of Monte-Carlo dropout masks

    Methods
    -------
    calculate_logits_val(model, dataloader, mixed_precision=False) : allows you to 
        get logits (outputs of the model) for further processing and estimation of 
        uncertainty according to the classical approach
    calculate_logits_mcd(model, dataloader, mixed_precision=False, n_masks=None) : 
        allows you to get logits for further processing and estimation of 
        uncertainty using the Monte Carlo dropout method
    calculate_classic_ue(methods_ue='all', topn=1, saving_plots=True, 
                         saving_roc_auc=True, draw_plots=False, logits=None, 
                         targets=None, path_saving_plots=None, metrics='accuracy', 
                         metrics_average='micro') : 
        allows you to get an estimate of uncertainty according to the classical 
        approach (entropy, maximum logits, maximum probability), an estimate of 
        roc_auc, as well as a graph for the metric
    calculate_mcd_ue(methods_ue='all', topn=1, saving_plots=True, 
                     saving_roc_auc=True, draw_plots=False, logits=None, 
                     targets=None, mcd_logits_list=None, path_saving_plots=None, 
                     metrics='accuracy', metrics_average='micro') : 
        allows you to get an estimate of uncertainty using the Monte-Carlo dropout 
        method (mean entropy, logits ratio, softmax ratio, probability variation, 
        variation ratio, bald score), an estimate of roc_auc, as well as a graph 
        for the metric
    """

    def __init__(self, conf):
        """
        Constructor of the Uncertainty class. 

        Parameters
        ----------
        conf : str or object
            configuration to run (the config itself or the path to it is passed)
            (config format .yaml or str - path to the config)
        """
        if isinstance(conf, str):
            with open(conf, "r") as stream:
                self.conf = yaml.safe_load(stream)
        else:
            self.conf = conf


    def __create_logits(self, model, dataloader, device):
        """
        Method for running data through the model and getting logits

        Parameters:
        ----------
        model : BertForSequenceClassification / another models
            A ready-made trained model, in which there are methods predict, eval, 
            train
        dataloader : torch.utils.data.dataloader.DataLoader / an iterable object 
                     whose elements are a dictionary {name: data}
            Used to load data into the model 
        device : torch.device
            the device on which the calculations will be performed

        Returns
        -------
        logits_list : list of shape(n_samples, n_logits)
            list of logit sets
        """
        logits_list = list()
        idx = 0
        for b_input_data in tqdm(iter(dataloader)):

            b_input_data = {name: data.to(device) for name, data in b_input_data.items()}

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        logits = model(**b_input_data)
                        logits = logits.detach().cpu()
                        logits = logits.numpy()
            else:
                with torch.no_grad():
                    logits = model(**b_input_data)
                    logits = logits.detach().cpu()
                    logits = logits.numpy()

            for logs in logits:
                logits_list.append(logs)

            idx += 1
        
        return logits_list


    def __converte_logits_to_array(self, logits_ds):
        """
        Converting a list of logit sets into an array for correct operation

        Parameters:
        ----------
        logits_ds : list or array of shape (n_samples, n_logits)
            list or array of logit sets

        Returns
        -------
        logits_list : array of shape (n_samples, n_logits)
            array of logit sets 
        """
        logits_list = list()
        for logits in logits_ds:
            logits_list.append(np.array(logits))
        logits_list = np.array(logits_list)
        return logits_list

    
    def __save_results(self, logits_array, mode='val'):
        """
        Method for saving results

        Parameters:
        ----------
        logits_array : array of shape (n_samples, n_logits) or 
                       shape(n_masks, n_samples, n_logits)
            array of logit sets
        mode : str, default=None
            save mode (classic / Monte Carlo dropout)

        Returns
        -------
        None 

        """
        saved_dict = self.conf['paths']['saved_files_paths']
        if mode == 'val':
            np.save(saved_dict['val_logits_path'] + saved_dict['names']['logits_val_name'], logits_array)
        elif mode == 'mcd':
            np.save(saved_dict['mcd_logits_path'] + saved_dict['names']['logits_mcd_all_name'], logits_array)    


    def __accuracy_topn(self, targets, preds, n=3):
        """
        Calculating the accuracy of the top n

        Parameters
        ----------
        targets : array of shape(n_samples, )
            array of targets
        preds : array of shape(n_samples, )
            array of predicts
        n : int, default=3
            the number of forecasts taken into account for the current sample (top n)

        Returns
        -------
        accuracy : float
            accuracy of top n
        """
        if n == 1:
            return np.count_nonzero(targets == preds) / len(targets)
        else:
            return np.count_nonzero(np.sum(np.broadcast_to(targets, (n, targets.shape[0])).T == preds, axis=1)) / len(targets)

   
    def __metrics_rejection(self, x, unc, y, preds):
        """
        Method for calculating accuracy with unc dropout (uncertainty)

        x, value, targets, preds

        Parameters
        ----------
        x : list of shape(n_points_by_ox, )
            a list with the number of samples being discarded
        unc : array of shape(n_samples, )
            an array with uncertainties for each sample
        y : array of shape(n_samples, )
            array of targets
        preds : array of shape(n_samples, )
            array of predicts

        Returns
        -------
        array of shape(x,)
            array with metric results on non-discarded samples
        """
        y = y.astype(np.int_)
        preds = preds.astype(np.int_)
        metrics = {
            'accuracy': lambda y_true, y_preds: self.__accuracy_topn(y_true, y_preds, n=self.topn),
            'f1': lambda y_true, y_preds: f1_score(y_true, y_preds, average=self.metrics_average),
            'recall': lambda y_true, y_preds: recall_score(y_true, y_preds, average=self.metrics_average),
            'precision': lambda y_true, y_preds: precision_score(y_true, y_preds, average=self.metrics_average)
        }

        rej_metrics_list = []
        for rej_samples in x:
            most_conf = np.argsort(unc)[:-rej_samples]
            rej_metrics = metrics[self.metrics](y[most_conf], preds[most_conf])
            rej_metrics_list.append(rej_metrics)
        return np.array(rej_metrics_list)


    def __entropy(self, x):
        """
        Entropy calculation based on the transmitted value

        Parameters
        ----------
        x : array or list of shape(n_logits, )
            a set of values for which entropy is to be calculated
        
        Returns
        -------
        float
            calculated entropy value
        """
        return np.sum(-x * np.log(np.clip(x, 1e-8, 1)), axis=-1)


    def __mean_entropy(self, sampled_probabilities):
        """
        Calculate the average entropy from the transmitted probabilities

        Parameters
        ----------
        sampled_probabilities : array or list of shape(n_samples, n_masks, n_logits)
            a set of values for which entropy is to be calculated
        
        Returns
        -------
        array of shape(n_samples, )
            calculated entropy values
        """
        return self.__entropy(np.mean(sampled_probabilities, axis=1))


    def __bald(self, sampled_probabilities):
        """
        Calculate the bald score based on the transmitted probabilities

        Parameters
        ----------
        sampled_probabilities : array or list of shape(n_samples, n_masks, n_logits)
            a set of values for which bald score is to be calculated
        
        Returns
        -------
        array of shape(n_samples, )
            calculated bald score values
        """
        predictive___entropy = self.__entropy(np.mean(sampled_probabilities, axis=1))
        expected___entropy = np.mean(self.__entropy(sampled_probabilities), axis=1)

        return predictive___entropy - expected___entropy


    def __probability_variance(self, sampled_probabilities, mean_probabilities=None):
        """
        Method for calculating probability variation

        Parameters
        ----------
        sampled_probabilities : array or list of shape(n_samples, n_masks, n_logits)
            a set of values for which probability variance is to be calculated
        mean_probabilities : array or list of shape(n_samples, n_logits), default=None
            a set of averaged probabilities for Monte-Carlo dropout masks

        Returns
        -------
        array of shape(n_samples, )
            calculated probability variance values

        """
        if mean_probabilities is None:
            mean_probabilities = np.mean(sampled_probabilities, axis=1)

        mean_probabilities = np.expand_dims(mean_probabilities, axis=1)

        return ((sampled_probabilities - mean_probabilities) ** 2).mean(1).sum(-1)


    def __var_ratio(self, sampled_probabilities):
        """
        Method for calculating varition ratio

        Parameters
        ----------
        sampled_probabilities : array or list of shape(n_samples, n_masks, n_logits)
            a set of values for which varition ratio is to be calculated

        Returns
        -------
        array of shape(n_samples, )
            calculated varition ratio values
        """
        top_classes = np.argmax(sampled_probabilities, axis=-1)
        # count how many time repeats the strongest class
        mode_count = lambda preds: np.max(np.bincount(preds))
        modes = [mode_count(point) for point in top_classes]
        ue = 1.0 - np.array(modes) / sampled_probabilities.shape[1]
        return ue


    def __n_largets(self, arr, idx=None, n=10):
        """
        Method for calculating and issuing top n maximum logits and their indexes

        Parameters
        ----------
        arr : array of shape(n_logits, )
            an array of logits of one sample
        idx : array of shape(n, ), default=None
            array of indexes of top n logits
        n : int, default=10
            number of top n logits

        Returns
        -------
        array of shape(n, )
            array of top n logits
        array of shape(n, )
            array of indexes of top n logits
        """
        if idx is None:
            idx = np.argsort(arr)[-n:].flatten()
        return arr[idx], idx


    def __get_index_topn(self, logits):
        """
        Calculation of top n indexes

        Parameters:
        ----------
        logits : array of shape(n_logits, )
            array of logits

        Returns
        -------
        array of shape(n, )
            indexes of top n logits
        """
        return np.asarray([self.__n_largets(np.array(l), n=self.topn)[1] for l in logits])

    
    def __get_logits_topn(self, logits, logits_idx_topn=None):
        """
        Calculation of top n logits

        Parameters:
        ----------
        logits : array of shape(n_logits, )
            array of logits
        logits_idx_topn : array of shape(n, ) of top n indexes, default=None
            indexes of top n logits

        Returns
        -------
        array of shape(n, )
            top n logits
        """
        if logits_idx_topn is None:
            logits_idx_topn = self.__get_index_topn(logits)
        return np.asarray([self.__n_largets(np.array(l), idx=logits_idx_topn[i])[0] for i, l in enumerate(logits)])


    def __compute_classic_uncertainty(self, logits, methods_ue: Union[List[str], str] = 'all'):
        """
        A method for calculating uncertainty according to the classical method

        Parameters:
        ----------
        logits : array of shape(n_sample, n_logits)
            array of logits
        methods_ue : list[str] or 'all', default='all'
            methods for estimating uncertainty

        Returns
        -------
        classic_ue_dict : {str: array of shape(n_samples, )} 
            a dictionary whose keys are methods, and whose values are an estimate of 
            uncertainty for all samples
        """
        if self.topn > 1:
            logits = self.__get_logits_topn(logits)

        methods = {
            'ent': lambda logits: np.array([self.__entropy(softmax(l)) for l in logits]),
            'max_prob': lambda logits: np.array([1 - np.max(softmax(l)) for l in logits]),
            'max_logits': lambda logits: np.array([1 - np.max(l) for l in logits])
        }

        assert methods_ue == 'all' or all([m in methods for m in methods_ue])

        if methods_ue == 'all':
            methods_ue = list(methods.keys())

        classic_ue_dict = dict()

        for method in methods_ue:
            ue = methods[method](logits)
            classic_ue_dict[method] = ue

        return classic_ue_dict


    def __get_roc_auc_dict(self, errors, ue_dict):
        """
        Calculation and issuance of roc_auc

        Parameters
        ----------
        errors : array or list of shape(n_samples, )
            array of prediction errors
        ue_dict : {str: array of shape(n_samples, )} 
            a dictionary whose keys are methods, and whose values are an estimate of 
            uncertainty for all samples

        Returns
        -------
        roc_auc_dict : {str: float}
            a dictionary whose keys are methods and whose values are roc auc score 
            for all samples
        """
        roc_auc_dict = dict()
        for key, value in ue_dict.items():
            roc_auc_dict[key] = roc_auc_score(errors, value)

        return roc_auc_dict


    def __compute_mcd_uncertainty(self, mcd_logits, methods_ue: Union[List[str], str] = 'all'):
        """
        A method for calculating uncertainty according to the Monte-Carlo dropout 
        method

        Parameters:
        ----------
        mcd_logits : array of shape(n_masks, n_samples, n_logits)
            array with sets of logits
        methods_ue : list[str] or 'all', default='all'
            methods for estimating uncertainty

        Returns
        -------
        mcd_ue_dict : {str: array of shape(n_samples, )} 
            a dictionary whose keys are methods, and whose values are an estimate of 
            uncertainty for all samples
        """
        methods = {
            'lr': lambda logs, probs: np.array([1 - np.max(l, axis=0) for l in logs.mean(0)]),
            'sr': lambda logs, probs: np.array([1 - np.max(l, axis=0) for l in probs.mean(0)]),
            'pv': lambda logs, probs: self.__probability_variance(probs.transpose(1, 0, 2)),
            'vr': lambda logs, probs: self.__var_ratio(probs.transpose(1, 0, 2)),
            'bald': lambda logs, probs: self.__bald(probs.transpose(1, 0, 2)),
            'mean_ent': lambda logs, probs: self.__mean_entropy(probs.transpose(1, 0, 2))
        }

        assert methods_ue == 'all' or all([m in methods for m in methods_ue])

        probs_mc = softmax(mcd_logits, axis=-1)

        if methods_ue == 'all':
            methods_ue = list(methods.keys())

        mcd_ue_dict = dict()

        for method in methods_ue:
            mc = methods[method](mcd_logits, probs_mc)
            mcd_ue_dict[method] = mc
        
        return mcd_ue_dict


    def __save_and_draw_plots(self, y_dict, targets, preds, xlabel='n_rejected', ylabel='accuracy_score', 
                            figsize=(10, 8), type_ue='classic'):
        """
        Method for drawing graphs and saving parameters

        Parameters
        ----------
        y_dict : {}
        targets : array or list of shape(n_samples, )
            array of targets
        preds : array or list of shape (n_samples, )
            array of predicts
        xlabel : str, default='n_rejected'
            x-axis caption
        ylabel : str, default='accuracy_score'
            y-axis caption
        figsize : (int, int), default=(10, 8)
            a tuple with the dimensions of the graph on the x and y axis
        type_ue : str, default='classic'
            the parameter to save is classic or Monte-Carlo dropout

        Returns
        -------
        None
        """
        # building plots
        x = range(10, len(targets), 1000)
        plt.figure(figsize=figsize)
        
        for key, value in y_dict.items():
            acc_y = self.__metrics_rejection(x, value, targets, preds)

            if self.topn > 1:
                plt.plot(x, acc_y, label=key + f'_top_{self.topn}')
            else:
                plt.plot(x, acc_y, label=key)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        
        # saving plots
        if self.saving_plots:
            saved_dict = self.conf['paths']['saved_files_paths']
            if self.path_saving_plots is None:
                saved_path = saved_dict['plots_path']
            else:
                saved_path = self.path_saving_plots

            if self.topn > 1:
                if type_ue == 'classic':
                    plt.savefig(saved_path + saved_dict['names']['plots_ue_classic_name'] + f'_top_{self.topn}' + '.png')
                elif type_ue == 'mcd':
                    plt.savefig(saved_path + saved_dict['names']['plots_ue_mcd_name'] + f'_top_{self.topn}' + '.png')
            else:
                if type_ue == 'classic':
                    plt.savefig(saved_path + saved_dict['names']['plots_ue_classic_name'] + '.png')
                elif type_ue == 'mcd':
                    plt.savefig(saved_path + saved_dict['names']['plots_ue_mcd_name'] + '.png')
        # drawing plots
        if not self.draw_plots:
            plt.close()


    def __save_roc_auc(self, roc_auc_dict, type_ue='classic'):
        """
        Saving roc auc

        Parameters
        ----------
        roc_auc_dict : {str: float}
            a dictionary whose keys are methods and whose values are roc auc score 
            for all samples
        type_ue : str, default='classic'
            the parameter to save is classic or Monte-Carlo dropout

        Returns
        -------
        None
        """
        saved_dict = self.conf['paths']['saved_files_paths']
        if self.topn > 1:
            if type_ue == 'classic':
                np.save(saved_dict['roc_auc_dict_path'] + saved_dict['names']['roc_auc_dict_classic_name'] + f'_top_{self.topn}', roc_auc_dict)
            elif type_ue == 'mcd':
                np.save(saved_dict['roc_auc_dict_path'] + saved_dict['names']['roc_auc_dict_mcd_name'] + f'_top_{self.topn}', roc_auc_dict)
        else:
            if type_ue == 'classic':
                np.save(saved_dict['roc_auc_dict_path'] + saved_dict['names']['roc_auc_dict_classic_name'], roc_auc_dict)
            elif type_ue == 'mcd':
                np.save(saved_dict['roc_auc_dict_path'] + saved_dict['names']['roc_auc_dict_mcd_name'], roc_auc_dict)
    

    def __get_logits_classic(self, model, val_dataloader):
        """
        The method that sets up and starts the calculation of logits

        Parameters
        ----------
        model : BertForSequenceClassification / another models
            A ready-made trained model, in which there are methods predict, eval, 
            train
        dataloader : torch.utils.data.dataloader.DataLoader / an iterable object 
                     whose elements are a dictionary {name: data}
            Used to load data into the model 

        Returns
        -------
        logits_list : list of shape(n_samples, n_logits)
            list of logit sets
        """
        device = torch.device(self.conf['device'])
        model.to(device)
        model.eval()
        logits_list = self.__create_logits(model, val_dataloader, device)

        return logits_list


    def __get_logits_mcd(self, model, dataloader):
        """
        The method that sets up and starts the calculation of Monte-Carlo dropout 
        logits

        Parameters
        ----------
        model : BertForSequenceClassification / another models
            A ready-made trained model, in which there are methods predict, eval, 
            train
        dataloader : torch.utils.data.dataloader.DataLoader / an iterable object 
                     whose elements are a dictionary {name: data}
            Used to load data into the model 

        Returns
        -------
        logits_list : list (n_masks, n_samples, n_logits)
            list of logit sets
        """
        device = torch.device(self.conf['device'])
        model.to(device)
        # to explicitly enable dropout
        model.train()
        mcd_res = list()
        if self.n_masks is None:
            N_MASKS = self.conf['monte_carlo_dropout_settings']['n_binary_masks']
        else:
            N_MASKS = self.n_masks
        for i in tqdm(range(N_MASKS)):
            _reset_mc_dropout(model, batch_idx=0, verbose=False)
            logits_ds = self.__create_logits(model, dataloader, device)
            logits_arr = self.__converte_logits_to_array(logits_ds)
            mcd_res.append(logits_arr)

        return np.asarray(mcd_res) 

    
    def __update_attributes(self, topn, saving_plots, saving_roc_auc, draw_plots, path_saving_plots, metrics, metrics_average):
        """        
        Updating attributes with passed values

        Parameters
        ----------
        topn : int
            the number of top forecasts taken into account in the calculation
        saving_plots : bool
            save the schedule or not
        saving_roc_auc : bool
            save the roc_auc score or not
        draw_plots : bool
            drawing a graph at once
        path_saving_plots : str
            the way to save graphs
        metrics : str
            the metric by which the method works
        metrics_average : str
            the type of averaging performed for the data

        Returns
        -------
        None
        """
        assert isinstance(topn, int) and topn > 0  
        assert isinstance(saving_plots, bool)
        assert isinstance(saving_roc_auc, bool)
        assert isinstance(draw_plots, bool)
        assert isinstance(path_saving_plots, str) or path_saving_plots is None
        assert metrics in ['accuracy', 'f1', 'recall', 'precision']
        assert metrics_average in ['binary', 'micro', 'macro', 'weighted', 'samples']

        self.topn = topn
        self.saving_plots = saving_plots
        self.saving_roc_auc = saving_roc_auc
        self.draw_plots = draw_plots
        self.path_saving_plots = path_saving_plots
        self.metrics = metrics
        self.metrics_average = metrics_average

    
    def calculate_logits_val(self, model, dataloader, mixed_precision=False):
        """
        This method calculates logic based on validation sampling, which
        are designed to estimate uncertainty by the classical method (entropy, 
        maximum logits, maximum probability) 

        Parameters:
        ----------
        model : BertForSequenceClassification / another models
            A ready-made trained model, in which there are methods predict, eval, 
            train
        dataloader : torch.utils.data.dataloader.DataLoader / an iterable object 
                     whose elements are a dictionary {name: data}
            Used to load data into the model
        mixed_precision : bool, default=False
            Mixed precision is the use of 16—bit and 32-bit floating point types in the 
            model during training so that it runs faster and uses less memory 

        Returns
        -------
        logits_array: ndarray of shape (n_samples, n_logits)
            A matrix of logits of the size n_samples (number of validation samples) 
            by n_logits (number of logits/outputs of the model)
        """
        assert isinstance(mixed_precision, bool)
        self.mixed_precision = mixed_precision
        # loading the model and passing the validation sample through the model to get the logits
        logits_ds = self.__get_logits_classic(model, dataloader)
        logits_list = self.__converte_logits_to_array(logits_ds)
        self.__save_results(logits_list, mode='val')

        return logits_list


    def calculate_logits_mcd(self, model, dataloader, mixed_precision=False, n_masks=None):
        """
        This method calculates logits based on a validation sample, but the output is 
        N different sets of logits. Next, these logits will be used to quantify 
        uncertainty using the Monte-Carlo dropout method (mean entropy, logits ratio, 
        softmax ratio, probability variation, variation ratio, bald score) 

        Parameters
        ----------
        model : BertForSequenceClassification / another models
            A ready-made trained model, in which there are methods predict, eval, 
            train.
        dataloader : torch.utils.data.dataloader.DataLoader / an iterable object 
                     whose elements are a dictionary {name: data}
            Used to load data into the model. 
        mixed_precision : bool, default=False
            Mixed precision is the use of 16—bit and 32-bit floating point types in the 
            model during training so that it runs faster and uses less memory.
        n_masks: int, default=None
            number of masks for Monte-Carlo dropout

        Returns
        -------
        logits_array: ndarray of shape (n_masks, n_samples, n_logits)
            Logits tensor of size n_masks (number of Monte Carlo dropout masks/number of 
            logit sets) by n_samples (number of validation samples) by n_logits (number 
            of model logits/outputs)
        """
        assert isinstance(mixed_precision, bool)
        assert isinstance(n_masks, int) or n_masks is None
        self.mixed_precision = mixed_precision
        self.n_masks = n_masks
        model_copy = copy.deepcopy(model)
        # convert the dropout layer
        _convert_dropouts(model_copy)
        # activate dropout
        _activate_mc_dropout(model_copy, activate=True, verbose=False)
        # reset the old MCDropout mask, create a new one
        _reset_mc_dropout(model_copy, batch_idx=0, verbose=False)
        mcd_res = self.__get_logits_mcd(model_copy, dataloader)
        self.__save_results(mcd_res, mode='mcd')
        
        return mcd_res
        
        
    def calculate_classic_ue(self, methods_ue='all', topn=1, saving_plots=True, saving_roc_auc=True, draw_plots=False, logits=None, 
                             targets=None, path_saving_plots=None, metrics='accuracy', metrics_average='micro'):
        """
        The method allows you to calculate the uncertainty estimate in the classical 
        way (entropy, maximum logits, maximum probability) for each sample, calculate 
        and estimate roc_auc, create a graph with uncertainty elimination based on the 
        transmitted metric

        Parameters
        ----------
        methods_ue: ['ent', 'max_prob', 'max_logits'] or 'all', default='all'
            Methods for calculating uncertainty estimates. This parameter is 
            responsible for which methods will be used to estimate the uncertainty
        topn: int, default=1
            This parameter specifies the number of top classes taken in one sample. 
            In this module, it is used only with the accuracy metric
        saving_plots: bool, default=True
            Save the schedule or not
        saving_roc_auc: bool, default=True
            Save the roc_auc characteristic or not
        draw_plots: bool, default=False
            Display graphs immediately after calculation or not
        logits: array of float of shape(n_samples, n_logits) or None, default=None
            Parameter for loading logits. When you enter None, the logits will be 
            loaded from the path defined in the config
        targets: array of int of shape(n_samples,) or None, default=None
            Parameter for loading targets (true classes). When you enter None, the 
            targets will be loaded from the path defined in the config
        path_saving_plots: str, default=None
            The path for saving graphs. When passing the path, the graphs will be 
            saved to the specified directory. If the path is specified as None, then 
            the graph saving paths will be taken from the config
        metrics: {'accuracy', 'f1', 'recall', 'precision'}, default='accuracy'
            The parameter specifies which metric to use for plotting graphs with 
            dropout based on uncertainty estimation. Only one metric can be passed
        metrics_average: {'micro', 'macro', 'weighted', 'binary', 'samples'}, 
                         default='micro'
            This parameter is required for multiclass/multilabel targets. This 
            determines the type of averaging performed on the data

        Returns
        -------
        classic_ue_dict, roc_auc_dict: dict({methods_ue: ue_array}), 
                                       dict({methods_ue: roc_auc})
            Returns two dictionaries. The first dictionary: keys are the names of the 
            methods that were used to calculate (str), values are floating-point arrays 
            with the form (n_samples,) with calculated uncertainty for each sample. The 
            second dictionary: the keys are the names of the methods that were used to 
            calculate (str), the values are float roc_auc.
        """
        self.__update_attributes(topn, saving_plots, saving_roc_auc, draw_plots, path_saving_plots, metrics, metrics_average)
        # loading logits, targets
        load_dict = self.conf['paths']['saved_files_paths']
        if logits is None:
            logits = np.load(load_dict['val_logits_path'] + load_dict['names']['logits_val_name'] + '.npy', allow_pickle=True)
        if targets is None:
            targets = np.load(load_dict['targets_path'] + load_dict['names']['targets_val_name'] + '.npy', allow_pickle=True)
        # since entropy outputs nan at values close to 0, you need to switch to the float32 type
        logits = logits.astype(np.float32) 
        preds = np.array([np.argmax(l, axis=0) for l in logits])
        errors = (targets != preds).astype(int)

        if self.topn > 1:
            preds = self.__get_index_topn(logits)
            logits = self.__get_logits_topn(logits, logits_idx_topn=preds)
        classic_ue_dict = self.__compute_classic_uncertainty(logits, methods_ue)
        roc_auc_dict = self.__get_roc_auc_dict(errors, classic_ue_dict)

        # save graphs and roc_auc
        if self.saving_plots or self.draw_plots:
            self.__save_and_draw_plots(classic_ue_dict, targets, preds, type_ue='classic', ylabel=self.metrics + '_score')
        if self.saving_roc_auc:
            self.__save_roc_auc(roc_auc_dict, type_ue='classic')

        return classic_ue_dict, roc_auc_dict
        

    def calculate_mcd_ue(self, methods_ue='all', topn=1, saving_plots=True, saving_roc_auc=True, draw_plots=False, 
                         logits=None, targets=None, mcd_logits_list=None, path_saving_plots=None, metrics='accuracy', metrics_average='micro'):
        """        
        The method allows you to calculate the uncertainty estimate by the Monte-Carlo 
        dropout method (mean entropy, logits ratio, softmax ratio, probability 
        variation, variation ratio, bald score) for each sample, calculate and 
        evaluate roc_auc, create a graph with the elimination of uncertainty based on 
        the transmitted metric
        
        Parameters
        ----------
        methods_ue: ['lr', 'sr', 'pv', 'vr', 'bald', 'mean_ent'] or 'all', 
                    default='all'
            Methods for calculating uncertainty estimates. This parameter is 
            responsible for which methods will be used to estimate the uncertainty
        topn: int, default=1
            This parameter specifies the number of top classes taken in one sample. 
            In this module, it is used only with the accuracy metric
        saving_plots: bool, default=True
            Save the schedule or not
        saving_roc_auc: bool, default=True
            Save the roc_auc characteristic or not
        draw_plots: bool, default=False
            Display graphs immediately after calculation or not
        logits: array of float of shape(n_samples, n_logits) or None, default=None
            Parameter for loading logits. When you enter None, the logits will be 
            loaded from the path defined in the config
        targets: array of int of shape(n_samples,) or None, default=None
            Parameter for loading targets (true classes). When you enter None, the 
            targets will be loaded from the path defined in the config
        mcd_logits_list: array of float of shape(n_binary_masks, n_samples, n_logits) 
                         or None, default=None
            Parameter for loading Monte Carlo dropout (MCD) logits. If you enter None, 
            the logits will be loaded along the path defined in the configuration
        path_saving_plots: str, default=None
            The path for saving graphs. When passing the path, the graphs will be 
            saved to the specified directory. If the path is specified as None, then 
            the graph saving paths will be taken from the config
        metrics: {'accuracy', 'f1', 'recall', 'precision'}, default='accuracy'
            The parameter specifies which metric to use for plotting graphs with 
            dropout based on uncertainty estimation. Only one metric can be passed
        metrics_average: {'micro', 'macro', 'weighted', 'binary', 'samples'}, 
                         default='micro'
            This parameter is required for multiclass/multilabel targets. 
            This determines the type of averaging performed on the data

        Returns
        -------
        mcd_ue_dict, roc_auc_dict: dict({methods_ue: ue_array}), 
                                   dict({methods_ue: roc_auc})
            Returns two dictionaries. The first dictionary: the keys (str) are the 
            names of the methods that were used for the calculation, the values are 
            float arrays with shape (n_samples,) with calculated uncertainty for each 
            sample. The second dictionary: the keys (str) are the names of the methods 
            that were used for the calculation , the values are float roc_auc.
        """
        self.__update_attributes(topn, saving_plots, saving_roc_auc, draw_plots, path_saving_plots, metrics, metrics_average)
        # loading logits, targets
        load_dict = self.conf['paths']['saved_files_paths']
        if mcd_logits_list is None:
            mcd_logits_list = np.load(load_dict['mcd_logits_path'] + load_dict['names']['logits_mcd_all_name'] + '.npy', allow_pickle=True)
        if logits is None:
            logits = np.load(load_dict['val_logits_path'] + load_dict['names']['logits_val_name'] + '.npy', allow_pickle=True)
        if targets is None:
            targets = np.load(load_dict['targets_path'] + load_dict['names']['targets_val_name'] + '.npy', allow_pickle=True)
        
        mcd_logits_list = np.array(mcd_logits_list).astype(np.float32)
        preds = np.array([np.argmax(l, axis=0) for l in logits])
        errors = (targets != preds).astype(int)

        if self.topn > 1:
            preds = np.asarray([self.__n_largets(np.array(l), n=topn)[1] for l in logits])

            mcd_logits_list_topn = []
            for mcd in mcd_logits_list:
                mcd_logits_list_topn.append(np.asarray([self.__n_largets(np.array(l), idx=preds[i])[0] for i, l in enumerate(mcd)]))
            mcd_logits_list = np.asarray(mcd_logits_list_topn)
            
        mcd_ue_dict = self.__compute_mcd_uncertainty(mcd_logits_list, methods_ue)
        roc_auc_dict = self.__get_roc_auc_dict(errors, mcd_ue_dict)
        
        # save and draw plots
        if self.saving_plots or self.draw_plots:
            self.__save_and_draw_plots(mcd_ue_dict, targets, preds, type_ue='mcd', ylabel=self.metrics + '_score')
        if self.saving_roc_auc:
            self.__save_roc_auc(roc_auc_dict, type_ue='mcd')

        return mcd_ue_dict, roc_auc_dict


class DropoutMC(torch.nn.Module):
    """
    The DropoutMC class the standard class of the torch.nn.Module, which contains 
    some methods for activating dropout Monte-Carlo, creating masks

    Atributes
    ---------
    activate : bool, default=False
        layer activation 
    p : float
        probability of layer loss
    p_init : float
        probability of layer loss with which we initialize
    batch_idx : int
        batch index
    binomial : torch.distrib
        binomial distribution from torch

    Methods
    -------
    _create_mask(input=None) : Method for creating a mask
    _truncate_mask(input, mask): Method for truncating the mask
    forward(x) : Method for forward
    """
    def __init__(self, p: float, activate=False):
        """
        Constructor of the DropoutMC class

        Parameters
        ----------
        p : float
            probability
        activate : bool, default=False
            layer activation        
        """
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p
        self.batch_idx = 0
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
        
    def _create_mask(
        self,
        input: torch.Tensor = None,
    ):  
        """
        Method for creating a mask

        Parameters
        ----------
        input : torch.Tensor, default=None
            the tensor of the layer to be masked

        Returns
        -------
        mask : torch.Tensor
            masking tensor
        """
        device = input.device
        shape = input.shape
        if len(input.shape) == 3:
            mask_shape = [1, 1, 2048]
        elif len(input.shape) == 4:
            mask_shape = [1, 1, 256, 256]
        elif len(input.shape) == 2:
            mask_shape = [1, 2048]
            
        mask = self.binomial.sample(torch.zeros(mask_shape).size()).to(device)
        return mask

    def _truncate_mask(self, input, mask):
        """
        Method for truncating the mask

        Parameters
        ----------
        input : torch.Tensor, default=None
            the tensor of the layer to be masked
        mask : torch.Tensor
            masking tensor

        Returns
        -------
        truncated_mask : torch.Tensor
            truncated tensor mask
        """
        slicers = [slice(0, 1)] + [slice(0, x) for x in input.shape[1:]]
        truncated_mask = mask[slicers]
        return truncated_mask

    def forward(self, x: torch.Tensor):
        """
        Method for forward

        Parameters
        ----------
        x : torch.Tensor
            tensor with data

        Returns
        -------
        torch.Tensor
            the result of applying a mask to a x weighted by the probability of 
            including the layer
        """
        if self.batch_idx == 0:
            self.mask = self._create_mask(x)
        self.batch_idx += 1
            
        mask = self._truncate_mask(x, self.mask)
        return x * mask / (1-self.p)
    

def _convert_to_mc_dropout(
    model: torch.nn.Module, substitution_dict: Dict[str, torch.nn.Module] = None
    ):
    """
    Convert dropout layers (Monte-Carlo dropout)
    
    Parameters
    ----------
    model : torch.nn.Module
        A ready-made trained model, in which there are methods predict, eval, 
        train
    substitution_dict : {str, torch.nn.Module]}, default=None
        dictionary with substitutions

    Returns
    -------
    None
    """
    for i, layer in enumerate(list(model.children())):
        proba_field_name = "dropout_rate" if "flair" in str(type(layer)) else "p"
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        proba_field_name = "drop_prob" if layer_name == "StableDropout" else proba_field_name
        if layer_name in substitution_dict.keys():
            model._modules[module_name] = substitution_dict[layer_name](
                p=getattr(layer, proba_field_name), activate=False
            )
        else:
            _convert_to_mc_dropout(model=layer, substitution_dict=substitution_dict)


def _activate_mc_dropout(
    model: torch.nn.Module, activate: bool, random: float = 0.0, verbose: bool = False
):
    """
    Activate dropout layers

    Parameters
    ----------
    model : torch.nn.Module
        A ready-made trained model, in which there are methods predict, eval, 
        train 
    activate: bool
        activation layer
    random : float, default=0.0
        probability of activation
    verbose : bool, default=False 
        display information about activated layers

    Returns
    -------
    None
    """
    for layer in model.children():
        if isinstance(layer, DropoutMC):
            if verbose:
                print(layer)
                print(f"Current DO state: {layer.activate}")
                print(f"Switching state to: {activate}")
            layer.activate = activate
            if activate and random:
                layer.p = random
            if not activate:
                layer.p = layer.p_init
        else:
            _activate_mc_dropout(
                model=layer, activate=activate, random=random, verbose=verbose
            )


def _reset_mc_dropout(
    model: torch.nn.Module, batch_idx: int, verbose: bool = False
):
    """
    Reset the old MCDropout mask and create a new one

    Parameters
    ----------
    model : torch.nn.Module
        A ready-made trained model, in which there are methods predict, eval, 
        train
    batch_idx: int
        butch index
    verbose : bool, default=False 
        display information about activated layers

    Returns
    -------
    None
    """
    for layer in model.children():
        if isinstance(layer, DropoutMC):
            if verbose:
                print(layer)
                print(f"Current DO state: {layer.batch_idx}")
                print(f"Switching state to: {batch_idx}")
            layer.batch_idx = batch_idx
        else:
            _reset_mc_dropout(
                model=layer, batch_idx=batch_idx, verbose=verbose
            )


def _convert_dropouts(model, dropout_subs='all'):
    """
    Convert dropout layers

    Parameters
    ----------
    model : torch.nn.Module
        A ready-made trained model, in which there are methods predict, eval, 
        train

    Returns
    -------
    None
    """
    dropout_ctor = lambda p, activate: DropoutMC(
        p=p, activate=False
    )

    # if dropout_subs == "last":
    #     _set_last_dropout(model, dropout_ctor(p=0.3, activate=False))
    
    if dropout_subs == "all":
        _convert_to_mc_dropout(model, {"Dropout": dropout_ctor, "StableDropout": dropout_ctor})

    else:
        raise ValueError(f"Wrong ue args")
