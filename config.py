class PipelineConfig:
    def __init__(self):
        self.seed = 59
        self.learning_rate = 1e-5
        self.epochs = 30
        self.train_batch_size = 16
        self.test_batch_size = 64
        self.hidden_dim = 512
        self.projection_dim = 512
        self.weight_decay = 1e-5
        self.patience = 3
        self.text_max_len = 50
        self.fusion_strategy = "concat+smalllen"
        self.text_encoder_id = "vinai/bartpho-word" # vinai/phobert-base-v2 # vinai/bartpho-word
        self.img_encoder_id = "timm/beitv2_base_patch16_224.in1k_ft_in22k" # timm/beitv2_base_patch16_224.in1k_ft_in22k # timm/resnet18.a1_in1k
        self.paraphraser_id = "chieunq/vietnamese-sentence-paraphase"
        self.is_text_augment = True
        self.n_text_paras = 2
        self.text_para_thresh = 0.8
        self.n_text_para_pool = 30
        self.start_threshold = 0.8
        self.min_threshold = 0.0
        self.use_dynamic_thresh = False
        self.restart_threshold = False
        self.restart_epoch = 0
        self.dataset_name = 'vivqa'
        self.data_dir = "/home/VLAI/datasets"
        self.use_amp = True


pipeline_config = PipelineConfig()
