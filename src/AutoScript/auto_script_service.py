from .auto_script_method import(Doe_sample_generate,Doe_execute)


def CreateSmpGenTask(smp_save_path:str,
                     param_ranges:dict[str, tuple[float, float]],
                     n_samples:int = 0,
                     level_nums:List[int] = [])->