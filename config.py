class PipelineConfig:
    def __init__(self):
        self.seed = 59
        self.learning_rate = 1e-5
        self.epochs = 30
        self.train_batch_size = 32
        self.test_batch_size = 64
        self.hidden_dim = 2048
        self.projection_dim = 2048
        self.weight_decay = 1e-5
        self.patience = 5
        self.text_max_len = 50
        self.fusion_strategy = "concat+smalllen"
        self.text_encoder_id = "vinai/bartpho-word"
        self.img_encoder_id = "timm/resnet18.a1_in1k"
        self.paraphraser_id = "chieunq/vietnamese-sentence-paraphase"
        self.is_text_augment = True
        self.num_paraphrase = 1
        self.paraphrase_thresh = 0.6
        self.n_para_pool = 10
        self.is_img_augment = False
        self.data_dir = "/home/VLAI/datasets"

pipeline_config = PipelineConfig()
