import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self, input_channels=None, seq_len=96, pred_len=96):
        super(BaseModel, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.pred_len = pred_len

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def _init_weights(self):
        """
        Initialize weights using Xavier/Glorot Uniform initialization
        similar to SAMFormer for consistency
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # TODO: get_experiment_summary might be more fitting
    # might also make sense to put this somewhere else
    def get_experiment_summary(self, args):
        """
        Generate formatted summary using all cli args
        """

        model_name = args.model_name

        width = 80
        top_border_with = width - 2

        lines = []

        # Top
        lines.append(" " + "=" * top_border_with)

        # Title
        title = f"MODEL CONFIGURATION: {model_name.upper()}"
        lines.append(title.center(top_border_with))
        lines.append(" " + "=" * top_border_with)

        # Parameter count
        param_info = f"Total Parameters: {self.param_num():,}"
        lines.append(param_info.center(width))
        lines.append(" " + "=" * top_border_with)

        arg_col_width = 36
        val_col_width = 37

        lines.append(f"| {'Argument'.ljust(arg_col_width)} | {'Value'.ljust(val_col_width)} |")
        lines.append(f"|{'-' * (arg_col_width + 3)}{'-' * (val_col_width + 2)}|")

        args_dict = vars(args)

        parser = args._parser

        if parser is not None:
            first_group = True
            for group in parser._action_groups:
                if group.title in [
                    "positional arguments",
                    "optional arguments",
                    "options",
                ]:
                    continue

                group_args = []
                for action in group._group_actions:
                    arg_name = action.dest
                    if arg_name in args_dict and arg_name != "help":
                        value = args_dict[arg_name]
                        group_args.append((arg_name, value))

                if group_args:
                    if not first_group:
                        lines.append(f"|{'-' * (arg_col_width + 3)}{'-' * (val_col_width + 2)}|")
                    first_group = False

                    group_header = f"*** {group.title} ***"
                    lines.append(f"| {group_header.ljust(arg_col_width + val_col_width + 3)} |")
                    lines.append(f"|{'-' * (arg_col_width + 3)}{'-' * (val_col_width + 2)}|")

                    for name, value in group_args:
                        if isinstance(value, bool):
                            value_str = "True" if value else "False"
                        elif value is None:
                            value_str = "None"
                        elif isinstance(value, (list, tuple)):
                            value_str = ", ".join(map(str, value))
                        else:
                            value_str = str(value)

                        if len(value_str) > val_col_width:
                            value_str = value_str[: val_col_width - 3] + "..."

                        lines.append(
                            f"| {name.ljust(arg_col_width)} | {value_str.ljust(val_col_width)} |"
                        )
        else:
            for name, value in sorted(args_dict.items()):
                # Format value
                if isinstance(value, bool):
                    value_str = "✓" if value else "✗"
                elif value is None:
                    value_str = "None"
                elif isinstance(value, (list, tuple)):
                    value_str = ", ".join(map(str, value))
                else:
                    value_str = str(value)

                if len(value_str) > val_col_width:
                    value_str = value_str[: val_col_width - 3] + "..."

                lines.append(f"| {name.ljust(arg_col_width)} | {value_str.ljust(val_col_width)} |")

        # Bottom border
        lines.append("=" * width)

        return "\n".join(lines)

    def print_experiment_summary(self, args, logger=None):
        """
        Print model summary.
        """
        summary = self.get_experiment_summary(args)

        if logger:
            logger.info("\n" + summary)
        else:
            print(summary)
