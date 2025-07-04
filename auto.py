import subprocess

def run_script(script_name):
    try:
        result = subprocess.run(['python', script_name], capture_output=True, text=True)
        print(f"\n--- {script_name} Output ---")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Error running {script_name}: {e}")

def main():
    scripts = [
        'certificate_watermark_embedding.py',
        'certificate_watermark_extraction.py',
        'noise.py',
        'psnr_ssim.py'
    ]

    for script in scripts:
        run_script(script)

if __name__ == '__main__':
    main()