def evaluate(program_path):
# 导入生成的程序
    import importlib.util
    import time

    spec = importlib.util.spec_from_file_location("module", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
# 测试性能

    start_time = time.time()
    result = module.optimize_function(100)
    execution_time=time.time() -start_time
    # accuracy=calculate_accuracy(result,expected_output)
# 返回多个评估指标
    return{
    # "accuracy":accuracy,
    "speed":1.0/(execution_time+0.001)
}
