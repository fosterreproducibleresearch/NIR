from ontolearn.knowledge_base import KnowledgeBase
import time
from typing import Iterable, Optional, Callable, Union, FrozenSet, Set, Dict, cast, Generator
from owlapy.abstracts import AbstractOWLOntology, AbstractOWLReasoner
from owlapy.owl_hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from owlapy.owl_individual import OWLNamedIndividual
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

from owlapy.render import DLSyntaxObjectRenderer
from owlapy.parser import DLSyntaxParser
from owlapy.utils import iter_count

from nir import InferenceDataset
from nir.config import NIRConfig
from nir.models import NIRComposite, NIRTransformer, NIRLSTM, NIRGRU
from nir.models.pmanet import PMAnet
from nir.utils import str2bool, read_embs_and_apply_agg
from nir.utils import score_all_inds, score_all_inds_composite


class NIRKB(KnowledgeBase):
    def __init__(self,
                 path: Optional[str] = None,
                 reasoner_factory: Optional[
                     Callable[[AbstractOWLOntology], AbstractOWLReasoner]] = None,
                 ontology: Optional[AbstractOWLOntology] = None,
                 reasoner: Optional[AbstractOWLReasoner] = None,
                 class_hierarchy: Optional[ClassHierarchy] = None,
                 load_class_hierarchy: bool = True,
                 object_property_hierarchy: Optional[ObjectPropertyHierarchy] = None,
                 data_property_hierarchy: Optional[DatatypePropertyHierarchy] = None,
                 include_implicit_individuals=False,
                 dataset_dir=None, model_list=None, model_paths=None, use_pma=True,
                 pma_model_path=None,
                 tokenizer_path=None,
                 chunksize=1024, th=0.5):

        self.th = th
        self.pma_net = None
        # self.model_name = model
        self.model_list = model_list
        self.model = "+".join(list(map(str.upper, model_list)))
        self.models = []
        self.chunksize = chunksize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model in model_list:
            if model.lower() == "composite" and use_pma:
                self.pma_net = PMAnet(NIRConfig().embedding_dim, NIRConfig().num_attention_heads,
                                      1)
                self.pma_net.load_state_dict(
                    torch.load(pma_model_path, map_location="cpu", weights_only=True)
                )
                self.pma_net.eval()

        self.kb, self.all_individuals, self.embeddings = read_embs_and_apply_agg(
            dataset_dir, nn_agg=self.pma_net, merge=True
        )
        self.dls_renderer = DLSyntaxObjectRenderer()
        self.all_individuals_set = set(self.all_individuals)
        self.all_individuals_arr = np.array(sorted(self.all_individuals), dtype=object)
        self.all_ind_embs = torch.FloatTensor(
            self.embeddings.loc[self.all_individuals_arr].values).to(self.device)

        kb_namespace = list(self.kb.ontology.classes_in_signature())[0].str
        if "#" in kb_namespace:
            self.kb_namespace = kb_namespace.split("#")[0] + "#"
        elif "/" in kb_namespace:
            self.kb_namespace = kb_namespace[:kb_namespace.rfind("/")] + "/"
        elif ":" in kb_namespace:
            self.kb_namespace = kb_namespace[:kb_namespace.rfind(":")] + ":"
        else:
            self.kb_namespace = kb_namespace
        self.expression_parser = DLSyntaxParser(self.kb_namespace)
        AutoConfig.register("nir", NIRConfig)

        for i in range(len(model_list)):
            model = model_list[i]
            if model.lower() == "composite":
                AutoModel.register(NIRConfig, NIRComposite)
            elif model.lower() == "lstm":
                AutoModel.register(NIRConfig, NIRLSTM)
            elif model.lower() == "gru":
                AutoModel.register(NIRConfig, NIRGRU)
            elif model.lower() == "transformer":
                AutoModel.register(NIRConfig, NIRTransformer)

            print("\n" + "\x1b[0;30;43m" + "Loading Model..." + "\x1b[0m" + "\n")
            self.models.append(AutoModel.from_pretrained(model_paths[i]).to(self.device))
            if model.lower() != "composite":
                self.tokenizer = AutoTokenizer.from_pretrained(model_paths[i])

        super().__init__(path=path,
                         reasoner_factory=reasoner_factory,
                         ontology=ontology,
                         reasoner=reasoner,
                         class_hierarchy=class_hierarchy,
                         load_class_hierarchy=load_class_hierarchy,
                         object_property_hierarchy=object_property_hierarchy,
                         data_property_hierarchy=data_property_hierarchy,
                         include_implicit_individuals=include_implicit_individuals)

    def nir_composite_predict(self, expr, model):
        start_time = time.time()
        if isinstance(expr, str):
            expr = [expr]

        data = InferenceDataset(
            data=expr,
            all_individuals=self.all_individuals_set,
            concept_to_instance_set=None,
            embeddings=self.embeddings
        )

        expr, component_embeddings_dict = next(iter(data))
        component_embeddings_dict = [
            {key: val.to(self.device) for key, val in component_embeddings_dict.items()}
        ]

        outputs = score_all_inds_composite(
            model, [expr], self.all_ind_embs, component_embeddings_dict
        ).squeeze()

        #retrieved = set(self.all_individuals_arr[np.where(outputs > self.th)[0]])
        time_taken = time.time() - start_time
        return outputs, time_taken

    def nir_encoders_predict(self, expr, model):
        start_time = time.time()
        if isinstance(expr, str):
            expr = [expr]
        outputs = score_all_inds(model, self.tokenizer, self.all_ind_embs, expr,
                                 hidden_size=self.all_ind_embs.shape[1],
                                 chunk_size=self.chunksize).squeeze()
        #retrieved = set(self.all_individuals_arr[np.where(outputs > self.th)[0]])
        time_taken = time.time() - start_time
        return outputs, time_taken

    def nir_predict(self, expr):
        result_dict = {}
        time_taken_dict = {}
        for model in self.model_list:
            pretrained_model = self.models[self.model_list.index(model)]
            if model.lower() == "composite":
                result_dict[model], time_taken_dict[model] = self.nir_composite_predict(expr,
                                                                                         pretrained_model)
            else:
                result_dict[model], time_taken_dict[model] = self.nir_encoders_predict(expr,
                                                                                        pretrained_model)
        # ensemble the models by taking mean of the values in result_dict
        mean_score = np.mean(list(result_dict.values()), axis=0)
        retrieved = set(self.all_individuals_arr[np.where(mean_score > self.th)[0]])
        return retrieved, time_taken_dict

    def individuals(self, concept=None, named_individuals=False):
        if concept:
            expr = concept if isinstance(concept, str) else self.dls_renderer.render(concept)
            individuals, _ = self.nir_predict(expr)
            # represent the individuals in kb namespace format
            namespace = self.kb_namespace.split("/", 3)
            namespace = "/".join(namespace[:3]) + "/"
            #print("namespace: ", namespace)
            individuals = list(map(lambda x: OWLNamedIndividual(namespace + x), individuals))

            return frozenset(individuals)

        else:
            return frozenset(self.ontology.individuals_in_signature())

    def __repr__(self):
        properties_count = iter_count(self.ontology.object_properties_in_signature()) + iter_count(
            self.ontology.data_properties_in_signature())
        class_count = iter_count(self.ontology.classes_in_signature())
        individuals_count = self.individuals_count()

        return f'NIRKB(path={repr(self.path)} [model={repr(self.model)}] <{class_count} classes, {properties_count} properties, ' \
               f'{individuals_count} individuals)>'