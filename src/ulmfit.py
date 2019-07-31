from fastai.text import *
import sentencepiece as spm
from fastai.callbacks import *
import pandas as pd

class ULMFiT():
    """docstring for ULMFiT"""
    def __init__(self, lang, path, finetune=True,
                 load_encoder=True, 
                 n_epocs_cls=100, 
                 n_epocs_lm=5, 
                 lr_cls=1e-2, 
                 lr_lm=1e-3, 
                 load_latest_checkpoint=False, 
                 model_name=None, 
                 train_lang_model = False,
                 df_lm = None):
        
        self.finetune_model = finetune
        self.load_encoder = load_encoder
        self.n_epocs_cls = n_epocs_cls
        self.n_epocs_lm = n_epocs_lm
        self.lr_cls = lr_cls
        self.lr_lm = lr_lm
        self.train_lang_model = train_lang_model
        self.load_latest_checkpoint=load_latest_checkpoint
        self.model_name = model_name
        self.lang = lang
        self.df_lm = df_lm
        self.tokenizer = Tokenizer(tok_func=LangTokenizer, lang=self.lang)
        self.data_lm = load_data(path/"ulmfit_models",f"data_{lang}_lm.pkl")
        self.path = path
        
    def load(self, df_train, df_test, df_lm):
        self.df_test = df_test
        self.df_lm = df_lm
        self.df_train = df_train[:-int(0.1*len(df_train))]
        print(f"valid : {-int(0.1*len(df_train))}")
        self.df_valid = df_train[-int(0.1*len(df_train)):]
        print(f"length of test : {len(self.df_test)}")
        print(f"Vocab Size : {data_lm.vocab}"
        self.data_cls = TextClasDataBunch.from_df(self.path/"ulmfit_training_data",
                                        train_df=self.df_train,
                                        test_df= self.df_test,
                                        valid_df= self.df_valid,
                                        tokenizer = self.tokenizer, 
                                        text_cols= "text", 
                                        label_cols= "label",
                                        vocab = self.data_lm.vocab)
        if self.train_lang_model:
            self.data_cls_lm = TextLMDataBunch.from_df(self.path/"ulmfit_training_data",
                                            train_df=self.df_lm,
                                            test_df= self.df_valid,
                                            valid_df= self.df_valid,
                                            tokenizer = self.tokenizer, 
                                            text_cols= "text", 
                                            label_cols= "label",
                                            vocab = self.data_lm.vocab)

        print("Data loaded successfully")
        
    def finetune(self, n_epocs=5, lr= 1e-3):
        config = awd_lstm_lm_config.copy()
        config['n_hid'] = 1152
        learn = language_model_learner(self.data_cls_lm, AWD_LSTM, config=config, drop_mult=0.5).to_fp16()
        print("Loading Wiki-Language Model")
        learn.load(f'{self.lang}-lm')
        print(f"Fitting LM for {n_epocs} epocs")
        learn.fit_one_cycle(n_epocs,lr)
        learn.save_encoder(f"{self.path}_{self.lang}_enc")
        print(f"Encoder Saved : {self.path}_{self.lang}_enc successfully")
        return f"{self.path}_{self.lang}_enc"
    
    def train_classifier(self, n_epocs=100, lr=1e-2, load_encoder= True, load_latest_checkpoint=False, model_name=None):
        learn = text_classifier_learner(self.data_cls, AWD_LSTM, drop_mult=0.5).to_fp16()
        if load_encoder:
            print("Loading Encoder")
            learn.load_encoder(f"{self.path}_{self.lang}_enc")
        if load_latest_checkpoint:
            print("Loading Checkpoint")
            try:
                self.load_trained_classifier(model_name)
                learn = self.learn
                print(f"Loaded {model_name} successfully")
            except:
                print("Can't load checkpoint, Enter Model Name")
        print(f"Fitting Classifier for {n_epocs} epocs")
        learn.fit_one_cycle(n_epocs, lr)
        print(f"Saving Classifier Model as :{self.lang}_{n_epocs}_epocs")
        learn.save(f'{self.lang}_{n_epocs}_epocs')
        return learn
    
    def load_trained_classifier(self, model_name):
        self.learn = text_classifier_learner(self.data_cls, AWD_LSTM, drop_mult=0.5)
        self.learn.load(model_name)
    
    def predict(self,text):
        return self.learn.predict(text)[0]
    
    def predict_proba(self,text):
        return self.learn.predict(text)[2]
 
    def fit(self,df_train, df_test):
        self.load(df_train, df_test, self.df_lm)
        if self.finetune_model:
            print("Finetuning the model")
            self.finetune(n_epocs=self.n_epocs_lm, lr = self.lr_lm)
        print("Training the classifier")
        learn = self.train_classifier(n_epocs=self.n_epocs_cls, 
                                      lr=self.lr_cls,
                                      load_encoder=self.load_encoder, 
                                      load_latest_checkpoint=self.load_latest_checkpoint,
                                      model_name=self.model_name)
        return learn
    
    def lm_data(self):
        self.data_lm.show_batch()
        return data_lm


class LangTokenizer(BaseTokenizer):
    def __init__(self, lang: str, vocab_size: int = 60000,  path_to_sp= "data/sentencepiece/"):
        self.lang = lang
        self.vocab_size = vocab_size
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(path_to_sp+f"{lang}_lm.model")
        self.vocab = Vocab([self.sp.IdToPiece(int(i)) for i in range(self.vocab_size)])

    def tokenizer(self, t: str) -> List[str]:
        return self.sp.EncodeAsPieces(t)

    def detokenizer(self, t: List[str]) -> str:
        return self.sp.DecodePieces(t)
