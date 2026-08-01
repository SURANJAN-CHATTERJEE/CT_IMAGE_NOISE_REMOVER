import sys
import json
import time

def run_smoke_tests():
    report = {"PassedModules": [], "FailedModules": [], "Errors": [], "Time": 0}
    start = time.time()
    
    stages = [
        ("Configuration Loading", "src.core.config", "ConfigManager"),
        ("Registry", "src.core.registry", "ExpertRegistry"),
        ("Hardware Monitor", "src.core.hardware", "HardwareMonitor"),
        ("Random Manager", "src.core.random_manager", "RandomManager"),
        ("DICOM/HU Pipeline", "src.pipeline.preprocessing.dicom.input_manager", ""),
        ("Synthetic Noise Generation", "src.pipeline.noise.noise_generator", ""),
        ("Adaptive Routing", "src.pipeline.routing.adaptive_router", ""),
        ("Expert Execution", "src.pipeline.restoration.expert_execution_manager", "ExpertExecutionManager"),
        ("Fusion", "src.pipeline.fusion.fusion_engine", "FusionEngine"),
        ("Explainability", "src.pipeline.explainability.explainability_engine", "ExplainabilityEngine"),
        ("Evaluation", "src.pipeline.evaluation.evaluation_engine", "EvaluationEngine"),
        ("Export", "src.pipeline.export.exporter", "OutputExporter"),
        ("Experiment Logging", "src.experiments.experiment_manager", "ExperimentManager")
    ]
    
    for name, module, cls_name in stages:
        try:
            mod = __import__(module, fromlist=[cls_name] if cls_name else ['*'])
            if cls_name:
                getattr(mod, cls_name)
            report["PassedModules"].append(name)
        except Exception as e:
            report["FailedModules"].append(name)
            report["Errors"].append(f"{name}: {str(e)}")
            
    report["Time"] = time.time() - start
    
    score = (len(report["PassedModules"]) / len(stages)) * 100
    
    with open("Validation_Report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    with open("Health_Score.json", "w") as f:
        json.dump({"HealthScore": score, "Passed": len(report["PassedModules"]), "Total": len(stages)}, f, indent=4)
        
    with open("Smoke_Test_Report.md", "w") as f:
        f.write("# Smoke Test Report\n\n")
        f.write(f"**Overall Health Score**: {score:.2f}%\n")
        f.write(f"**Execution Time**: {report['Time']:.4f}s\n\n")
        f.write("## Passed Modules\n")
        for m in report["PassedModules"]:
            f.write(f"- [x] {m}\n")
        f.write("\n## Failed Modules\n")
        for m in report["FailedModules"]:
            f.write(f"- [ ] {m}\n")
        f.write("\n## Errors\n")
        for e in report["Errors"]:
            f.write(f"- {e}\n")
            
    print(f"Health Score: {score:.2f}%")
    print(f"Failed Modules: {len(report['FailedModules'])}")
    for e in report["Errors"]:
        print(f"Error: {e}")
            
if __name__ == "__main__":
    run_smoke_tests()
