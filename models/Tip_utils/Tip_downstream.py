import torch
import torch.nn as nn
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
import sys
from omegaconf import DictConfig, open_dict, OmegaConf
# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('path/to/your/project')

from models.Tip_utils.Transformer import TabularTransformerEncoder, MultimodalTransformerEncoder
from models.Tip_utils.pieces import DotDict
from models.Tip_utils.VisionTransformer_imagenet import create_vit

class TIPBackbone(nn.Module):
  """
  Evaluation model for TIP.
  """
  def __init__(self, args) -> None:
    super(TIPBackbone, self).__init__()
    self.missing_tabular = args.missing_tabular
    print(f'Current missing tabular for TIPBackbone: {self.missing_tabular}')
    if args.checkpoint:
      print(f'Checkpoint name: {args.checkpoint}')

      checkpoint = torch.load(args.checkpoint)
      original_args = OmegaConf.create(checkpoint['hyper_parameters'])
      original_args.field_lengths_tabular = args.field_lengths_tabular
      state_dict = checkpoint['state_dict']
      if 'algorithm_name' not in original_args:
        with open_dict(original_args):
          original_args.algorithm_name = args.algorithm_name

      self.hidden_dim = original_args.multimodal_embedding_dim

      if 'encoder_imaging.0.weight' in state_dict:
        self.encoder_name_imaging = 'encoder_imaging.'

      else:
        encoder_name_dict = {'clip' : 'encoder_imaging.', 'remove_fn' : 'encoder_imaging.', 'supcon' : 'encoder_imaging.', 'byol': 'online_network.encoder.', 'simsiam': 'online_network.encoder.', 'swav': 'model.', 'barlowtwins': 'network.encoder.'}
        self.encoder_name_imaging = encoder_name_dict[original_args.loss]

      if original_args.model.startswith('vit'):
        self.encoder_imaging = create_vit(original_args)

      elif original_args.model.startswith('resnet'):
        self.encoder_imaging = torchvision_ssl_encoder(original_args.model, return_all_feature_maps=True)
      
      self.create_tabular_model(original_args)
      self.encoder_name_tabular = 'encoder_tabular.'
      assert len(self.cat_lengths_tabular) == original_args.num_cat
      assert len(self.con_lengths_tabular) == original_args.num_con

      self.create_multimodal_model(original_args)
      self.encoder_name_multimodal = 'encoder_multimodal.'

      for module, module_name in zip([self.encoder_imaging, self.encoder_tabular, self.encoder_multimodal], 
                                     [self.encoder_name_imaging, self.encoder_name_tabular, self.encoder_name_multimodal]):
        self.load_weights(module, module_name, state_dict)
        if args.finetune_strategy == 'frozen':
          for _, param in module.named_parameters():
            param.requires_grad = False

          parameters = list(filter(lambda p: p.requires_grad, module.parameters()))
          assert len(parameters)==0
          print(f'Freeze {module_name}')

        elif args.finetune_strategy == 'trainable':
          print(f'Full finetune {module_name}')

        else:
          assert False, f'Unknown finetune strategy {args.finetune_strategy}'

    else:
      self.create_imaging_model(args)
      self.create_tabular_model(args)
      self.create_multimodal_model(args)
      self.hidden_dim = args.multimodal_embedding_dim

    self.classifier = nn.Linear(self.hidden_dim, args.num_classes)

    if hasattr(args, "num_masked") and args.num_masked:
        self.mask_pred_head = nn.Linear(self.hidden_dim, args.num_masked)

    else:
        self.mask_pred_head = None

  def create_imaging_model(self, args):
    if args.model.startswith('vit'):
      self.encoder_imaging = create_vit(args)

    elif args.model.startswith('resnet'):
      self.encoder_imaging = torchvision_ssl_encoder(args.model, return_all_feature_maps=True)
  
  def create_tabular_model(self, args):
    self.field_lengths_tabular = torch.load(args.field_lengths_tabular)
    self.cat_lengths_tabular = []
    self.con_lengths_tabular = []
    for x in self.field_lengths_tabular:
      if x == 1:
        self.con_lengths_tabular.append(x) 

      else:
        self.cat_lengths_tabular.append(x)

    self.encoder_tabular = TabularTransformerEncoder(args, self.cat_lengths_tabular, self.con_lengths_tabular)

   
  def create_multimodal_model(self, args):
    self.encoder_multimodal = MultimodalTransformerEncoder(args)
  
  def load_weights(self, module, module_name, state_dict):
    state_dict_module = {}
    for k in list(state_dict.keys()):
      if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
        state_dict_module[k[len(module_name):]] = state_dict[k]

    print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
    log = module.load_state_dict(state_dict_module, strict=True)
    assert len(log.missing_keys) == 0

  def forward(self, x: torch.Tensor, visualize=False) -> torch.Tensor:
    x_i, x_t = x[0], x[1]
    x_i = self.encoder_imaging(x_i)[-1]
    if self.missing_tabular:
      missing_mask = x[2]
      x_t = self.encoder_tabular(x=x_t, mask=missing_mask, mask_special=missing_mask)

    else:
      x_t = self.encoder_tabular(x_t)
      
    if visualize==False:
      x_m = self.encoder_multimodal(x=x_t, image_features=x_i)

    else:
      x_m, attn = self.encoder_multimodal(x=x_t, image_features=x_i, visualize=visualize)

    sales_output = self.classifier(x_m[:,0,:])

    if self.mask_pred_head is not None:
        masked_output = self.mask_pred_head(x_m[:,0,:])

    else:
        masked_output = None

    if visualize==False:
      return sales_output, x_m, masked_output
      
    else:
      return sales_output, (x_m, attn, masked_output)