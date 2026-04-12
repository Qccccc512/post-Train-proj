from huggingface_hub import HfApi

from hf_repo_sync import build_api, load_hf_config

def main():
    api, _, repo_id = build_api(load_hf_config('configs/hf/default.yaml'))
    run_name = "stage2search_20260407_155628_stage2_search_lr1e4_r16_e1_ms500"
    repo_path = f"runs/{run_name}"
    
    print(f"Deleting {repo_path} from {repo_id}...")
    try:
        api.delete_folder(repo_id=repo_id, repo_type="model", path_in_repo=repo_path)
        print("Successfully deleted.")
    except Exception as e:
        print(f"Failed to delete: {e}")

if __name__ == "__main__":
    main()
