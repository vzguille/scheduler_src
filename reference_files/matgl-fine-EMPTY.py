import pickle
from matgl import load_model
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.utils.training import PotentialLightningModule
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes

from dgl.data.utils import Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger


with open('../all_static_labels.pkl', 'rb') as f:
    all_static_labels = pickle.load(f)
with open('../all_static_structures.pkl', 'rb') as f:
    all_static_structures = pickle.load(f)
with open('../train_static_index.pkl', 'rb') as f:
    train_static_index = pickle.load(f)
with open('../test_static_index.pkl', 'rb') as f:
    test_static_index = pickle.load(f)
with open('../val_static_index.pkl', 'rb') as f:
    val_static_index = pickle.load(f)


element_types = get_element_list(all_static_structures)

converter = Structure2Graph(element_types=element_types, cutoff=5.0)
dataset = MGLDataset(
    threebody_cutoff=4.0,
    structures=all_static_structures,
    converter=converter,
    labels=all_static_labels,
)
train_data = Subset(
    dataset,
    train_static_index
)
val_data = Subset(
    dataset,
    val_static_index
)
test_data = Subset(
    dataset,
    test_static_index
)

train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_pes,
    batch_size!VAR,
    num_workers!VAR,
)

m3gnet_nnp = load_model("M3GNet-MP-2021.2.8-PES")
model = m3gnet_nnp.model

lit_module = PotentialLightningModule(model, 
    lr!VAR,
    energy_weight!VAR, 
    force_weight!VAR, 
    stress_weight!VAR,
    )

logger = CSVLogger("logs", 
                   name!VAR
                   )

trainer = pl.Trainer(max_epochs!VAR,
                     logger=logger,
                     inference_mode=False,
                     )  # initializing trainer

trainer.fit(model=lit_module, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader) # we use the module and the initialized trainer

model_export_path = "./model/"
model.save(model_export_path)
