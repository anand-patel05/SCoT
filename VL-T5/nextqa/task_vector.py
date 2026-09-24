import torch
import re
from collections import defaultdict


def fix_key_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module.vis_encoder."):
            new_k = "module.encoder." + k[len("module.vis_encoder."):]
        elif k.startswith("vis_encoder."):
            new_k = "encoder." + k[len("vis_encoder."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


def extract_layer_id(param_name):
    """
    Returns a unique layer id for encoder and decoder blocks.
    """
    match = re.search(r"(encoder|decoder)\.block\.(\d+)\.", param_name)
    if match:
        module = match.group(1)      # encoder or decoder
        idx = int(match.group(2))
        return f"{module}_{idx}"     # unique key
    return "shared"


def is_linear_param(name: str):
    """Return whether ``name`` identifies an affine VL-T5 weight."""
    if not name.endswith(".weight"):
        return False
    if "layer_norm" in name or "norm" in name:
        return False
    if "embed" in name or "embedding" in name:
        return False
    if "relative_attention_bias" in name:
        return False
    if "shared.weight" in name:
        return False
    return True


def depth_stability_cap(name, layer_id, alpha_min=0.08, alpha_max=0.45):
    """Return the maximum mixing weight allowed for a model block."""
    match = re.fullmatch(r"(encoder|decoder)_(\d+)", str(layer_id))
    if match is None:
        return alpha_max

    module, block_idx = match.group(1), int(match.group(2))

    if module == "encoder" and "encoder.block" in name:
        if 8 <= block_idx <= 11:    # late encoder  (8 - 11)
            return alpha_min
        return alpha_max

    if module == "decoder" and "EncDecAttention" in name:
        if block_idx <= 3:          # early decoder  (0 - 3)
            return min(alpha_min * 1.5, alpha_max)
        return alpha_max

    return alpha_max


class TaskVector:
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        if vector is not None:
            if isinstance(vector, dict):
                self.vector = fix_key_prefix(vector)
            else:
                self.vector = fix_key_prefix(torch.load(vector, map_location="cpu"))
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                w0 = fix_key_prefix(torch.load(pretrained_checkpoint, map_location="cpu"))
                wt = fix_key_prefix(torch.load(finetuned_checkpoint, map_location="cpu"))

                self.vector = {}
                for k in w0:
                    if k in wt and w0[k].dtype.is_floating_point:
                        self.vector[k] = wt[k] - w0[k]

    # --------------------------------------------------
    # Basic vector algebra (unchanged)
    # --------------------------------------------------
    def __add__(self, other):
        with torch.no_grad():
            new_vec = {}
            for k in self.vector:
                if k in other.vector:
                    new_vec[k] = self.vector[k] + other.vector[k]
        return TaskVector(vector=new_vec)

    def __neg__(self):
        with torch.no_grad():
            new_vec = {k: -v for k, v in self.vector.items()}
        return TaskVector(vector=new_vec)

    def scale(self, alpha: float):
        with torch.no_grad():
            new_vec = {k: alpha * v for k, v in self.vector.items()}
        return TaskVector(vector=new_vec)

    # --------------------------------------------------
    # Geometry utilities (global)
    # --------------------------------------------------
    def dot(self, other):
        s = 0.0
        for k in self.vector:
            if k in other.vector:
                s += torch.sum(self.vector[k] * other.vector[k])
        return s

    def norm(self):
        total = 0.0
        for v in self.vector.values():
            total += torch.sum(v.float() ** 2)
        return torch.sqrt(total)

    def project_away(self, other, eps=1e-8):
        denom = other.dot(other) + eps
        coef = self.dot(other) / denom
        new_vec = {}
        for k in self.vector:
            new_vec[k] = self.vector[k] - coef * other.vector[k]
        return TaskVector(vector=new_vec)

    # --------------------------------------------------
    # NEW: Layer-wise utilities (THIS IS THE UPDATE)
    # --------------------------------------------------
    def group_by_layer(self):
        """
        Returns:
            dict[layer_id -> dict[param_name -> tensor]]
        """
        layers = defaultdict(dict)
        for k, v in self.vector.items():
            layer_id = extract_layer_id(k)
            layers[layer_id][k] = v
        return layers

    def get_layer_vector(self, layer_id):
        """
        Returns a TaskVector containing only parameters from layer_id
        """
        sub_vec = {
            k: v for k, v in self.vector.items()
            if extract_layer_id(k) == layer_id
        }
        return TaskVector(vector=sub_vec)

    def layer_norms(self):
        """
        Returns:
            dict[layer_id -> L2 norm]
        """
        norms = {}
        for lid, params in self.group_by_layer().items():
            total = 0.0
            for v in params.values():
                total += torch.sum(v.float() ** 2)
            norms[lid] = torch.sqrt(total)
        return norms

    def layer_dot(self, other):
        """
        Returns:
            dict[layer_id -> dot product]
        """
        dots = {}
        self_layers = self.group_by_layer()
        other_layers = other.group_by_layer()

        for lid in self_layers:
            if lid in other_layers:
                s = 0.0
                for k in self_layers[lid]:
                    if k in other_layers[lid]:
                        s += torch.sum(
                            self_layers[lid][k] * other_layers[lid][k]
                        )
                dots[lid] = s
        return dots

    def layer_project_away(self, other, eps=1e-8):
        """
        Performs projection independently per layer.
        """
        new_vec = {}
        self_layers = self.group_by_layer()
        other_layers = other.group_by_layer()

        for lid in self_layers:
            if lid not in other_layers:
                new_vec.update(self_layers[lid])
                continue

            dot = 0.0
            norm2 = 0.0
            for k in self_layers[lid]:
                if k in other_layers[lid]:
                    dot += torch.sum(
                        self_layers[lid][k] * other_layers[lid][k]
                    )
                    norm2 += torch.sum(
                        other_layers[lid][k] ** 2
                    )

            coef = dot / (norm2 + eps)

            for k in self_layers[lid]:
                new_vec[k] = self_layers[lid][k] - coef * other_layers[lid][k]

        return TaskVector(vector=new_vec)

    # --------------------------------------------------
    # Application (unchanged)
    # --------------------------------------------------
    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        with torch.no_grad():
            base = fix_key_prefix(torch.load(pretrained_checkpoint, map_location="cpu"))
            new_state = {}

            for k in base:
                if k in self.vector:
                    vec = self.vector[k].to(
                        device=base[k].device,
                        dtype=base[k].dtype
                    )
                    coef = torch.tensor(
                        scaling_coef,
                        device=base[k].device,
                        dtype=base[k].dtype
                    )
                    new_state[k] = base[k] + coef * vec

        return new_state


def resolve_linear_conflict(current, previous, has_conflict, eps=1e-8):
    """Return current linear parameters, projected only when they conflict."""
    current_linear = {
        key: value for key, value in current.vector.items()
        if is_linear_param(key)
    }
    if not has_conflict:
        return TaskVector(vector=current_linear)

    overlapping_keys = [
        key for key in current_linear
        if key in previous.vector and is_linear_param(key)
    ]
    if not overlapping_keys:
        return TaskVector(vector=current_linear)

    dot = sum(
        torch.sum(current_linear[key] * previous.vector[key])
        for key in overlapping_keys
    )
    norm2 = sum(
        torch.sum(previous.vector[key].float() ** 2)
        for key in overlapping_keys
    )
    coef = dot / (norm2 + eps)

    projected = {}
    for key, value in current_linear.items():
        if key in previous.vector:
            projected[key] = value - coef * previous.vector[key]
        else:
            projected[key] = value
    return TaskVector(vector=projected)


def consolidate_layer(previous, current, layer_id, similarity,
                      alpha_min=0.08, alpha_max=0.45):
    """Consolidate one layer while projecting only its linear parameters."""
    alpha_l = alpha_min + (alpha_max - alpha_min) * (similarity + 1.0) / 2.0
    alpha_l = min(max(float(alpha_l), alpha_min), alpha_max)

    effective_linear = resolve_linear_conflict(
        current,
        previous,
        has_conflict=similarity < 0,
    )

    consolidated = {}
    for key, current_value in current.vector.items():
        if is_linear_param(key):
            cap = depth_stability_cap(
                key, layer_id, alpha_min=alpha_min, alpha_max=alpha_max
            )
            alpha_eff = min(alpha_l, cap)
            previous_value = previous.vector.get(key, 0.0)
            consolidated[key] = (
                (1.0 - alpha_eff) * previous_value
                + alpha_eff * effective_linear.vector[key]
            )
        elif key in previous.vector:
            # Nonlinear/infrastructure parameters are never projected.
            consolidated[key] = 0.5 * (
                previous.vector[key] + current_value
            )
        else:
            consolidated[key] = current_value

    return consolidated


def merge_max_abs(task_vectors):
    """Mix multiple task vectors together by highest parameter value."""
    if len(task_vectors) == 0:
        return task_vectors[0]
        
    with torch.no_grad():
        new_vector = {}
        
        # Iterate over keys in the first task vector
        for key in task_vectors[0].vector:
            # Get the initial tensor for the current key
            max_abs_tensor = task_vectors[0].vector[key]
            
            # Iterate over the remaining task vectors
            for task_vector in task_vectors[1:]:
                current_tensor = task_vector.vector[key]
                
                # Update max_abs_tensor to keep the element-wise maximum absolute values
                max_abs_tensor = torch.where(current_tensor.abs() >= max_abs_tensor.abs(), current_tensor, max_abs_tensor)
            
            # Assign the final tensor to the new_vector dictionary
            new_vector[key] = max_abs_tensor

    return TaskVector(vector=new_vector)
