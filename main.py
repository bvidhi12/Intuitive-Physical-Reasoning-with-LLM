import json
import subprocess
import resource

from datasets import load_dataset
from tqdm import tqdm
from multiprocess.pool import Pool


def format_story(sentences):
    result = ""
    for i, sentence in enumerate(sentences):
        result += f"{i + 1}. {sentence} "
    return result


def format_prompt_baseline(story):
    prompt = "### Given the following two stories, which one is more physically unlikely? Answer as [X, Y] where X is the unlikely story while Y is the unlikely sentence in that story. "
    story_a = story["stories"][0]
    story_b = story["stories"][1]

    prompt += ("### Story A: " + format_story(story_a["sentences"]) +
               "### Story B: " + format_story(story_b["sentences"]))
    prompt += "### Answer: "
    return prompt


def parse_response(response: str):
    response = response.split("### Answer: [")
    if len(response) != 2:
        print(f"Bad response: {response}")
        return {}

    response = response[1]
    return {
        "story": response[0],
        "sentence": int(response[3]) - 1,
        "reason": response[14:]
    }

def save_list(name, value):
    with open(name, "w") as file:
        for result in value:
            file.write(json.dumps(result) + "\n")

def load_list(name) -> list[dict]:
    with open(name, "r") as file:
        return [json.loads(line) for line in file]

def run_raw(command, model, prompt, grammar=None, length=32):
    command = [command, "-m", model, "-p", prompt, "-n", str(length), "-t", "1", "-c", "2048"]

    if grammar is not None:
        command.append("--grammar-file")
        command.append(grammar)

    before = resource.getrusage(resource.RUSAGE_CHILDREN)

    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)

    delta_time = after.ru_utime - before.ru_utime
    output = result.stdout.decode("utf-8")
    return {"response": output, "time": delta_time}

def run_baseline(command, model, prompt):

    result = run_raw(command, model, prompt, "grammar.gbnf")
    result |= parse_response(result["response"])
    del result["response"]
    return result


def run_single(command, model, story):

    prompt = format_prompt_baseline(story)
    result = run_baseline(command, model, prompt)

    result["id"] = story["example_id"]
    result["correct_story"] = "A" if story["label"] == 0 else "B"
    result["correct_sentence"] = story["confl_sents"]

    return result

def run_llama(story):

    command = "llama.cpp/build/bin/llama-cli"
    model = "llama.cpp/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
    return run_single(command, model, story)


def run_bitnet(story):

    command = "BitNet/build/bin/llama-cli"
    model = "BitNet/models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf"
    return run_single(command, model, story)

def run():

    dataset = load_dataset("sled-umich/TRIP")
    dataset = dataset["OrderTest"]
    # dataset = dataset.shuffle()
    dataset = dataset.take(100)

    prompt = format_prompt_baseline(dataset[0])
    print(prompt)
    return

    with Pool(processes=20) as pool:
        results = list(tqdm(pool.imap_unordered(run_bitnet, dataset, chunksize=1), total=len(dataset)))

    save_list("output-bitnet.txt", results)

    with Pool(processes=20) as pool:
        results = list(tqdm(pool.imap_unordered(run_llama, dataset, chunksize=1), total=len(dataset)))

    save_list("output-llama.txt", results)

def evaluate_result(name):

    results = load_list(name)

    count_story = list(result["story"] == result["correct_story"] for result in results).count(True)
    count_sentence = list(result["sentence"] in result["correct_sentence"] for result in results).count(True)

    import statistics
    mean = statistics.fmean(float(result["time"]) for result in results)
    print(f"For {name} there are {count_story} / {len(results)} correct stories and {count_sentence} / {len(results)} correct reasoning sentences with {mean} seconds of average inference time.")


def test():

    evaluate_result("output-bitnet.txt")
    evaluate_result("output-llama.txt")


def main():

    run()
    # test()

if __name__ == '__main__':
    main()
