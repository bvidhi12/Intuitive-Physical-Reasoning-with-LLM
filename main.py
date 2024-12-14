import json
import subprocess
import resource

from datasets import load_dataset
from tqdm import tqdm
from multiprocess.pool import Pool


def format_sentences(sentences: list[str]) -> str:
    result = ""
    for i, sentence in enumerate(sentences):
        result += f"{i + 1}. {sentence} "
    return result


def format_story(story, prefix) -> str:
    prompt = ""
    prompt += f"### {prefix} A: {format_sentences(story["stories"][0]["sentences"])}"
    prompt += f"### {prefix} B: {format_sentences(story["stories"][1]["sentences"])}"
    return prompt


def get_conflicting(story) -> list[int]:
    return [value + 1 for value in [*story["confl_sents"], story["breakpoint"]]]


def get_story_info(story) -> dict:
    result = {"id": story["example_id"],
              "correct_story": "B" if story["label"] == 0 else "A",  # The unlikely/implausible story
              "correct_sentence": get_conflicting(story)}
    return result


def save_list(name: str, value: list[dict]) -> None:
    with open(name, "w") as file:
        for result in value:
            file.write(json.dumps(result) + "\n")


def load_list(name: str) -> list[dict]:
    try:
        with open(name, "r") as file:
            return [json.loads(line) for line in file]
    except FileNotFoundError:
        return []


def run_raw(command: str, model: str, prompt: str, grammar=None, length=32) -> (str, float):
    command = [command, "-m", model, "-p", prompt, "-n", str(length), "-t", "1", "-c", "2048"]

    if grammar is not None:
        command.append("--grammar-file")
        command.append(grammar)

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)

    delta_time = after.ru_utime - before.ru_utime
    output = result.stdout.decode("utf-8")
    return output, delta_time


def pool_execute(dataset, action):
    import time
    before = time.monotonic()

    with Pool(processes=20) as pool:
        iterator = tqdm(pool.imap_unordered(action, dataset, chunksize=1), total=len(dataset))
        results = list(iterator)

    after = time.monotonic()
    print(f"Pool execution took {after - before} seconds in real time.")
    return results


def run_llama(prompt: str, grammar=None, length=32) -> (str, float):
    command = "llama.cpp/build/bin/llama-cli"
    model = "llama.cpp/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
    return run_raw(command, model, prompt, grammar, length)


def run_bitnet(prompt: str, grammar=None, length=32) -> (str, float):
    command = "BitNet/build/bin/llama-cli"
    model = "BitNet/models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf"
    return run_raw(command, model, prompt, grammar, length)


def baseline_run(runner, story):
    prompt = "### Given the following two stories, which one is more physically unlikely? "
    prompt += "Answer as [X, Y] where X is the unlikely story while Y is the unlikely sentence in that story. "
    prompt += format_story(story, "Story") + "### Answer: "
    output, time = runner(prompt, "grammar/baseline.gbnf", length=32)
    print(output)

    try:
        response = output.split("### Answer: [")
        answer = response[1]
        result = get_story_info(story) | {
            "story": answer[0],
            "sentence": int(answer[3]),
            "reason": answer[14:],
            "time": time
        }
    except Exception:
        print(f"@@@ Bad output: {output}")
        return {}

    return result


def reasoning_run_impl(runner, story, prompt):
    chain_of_thought = "### If we consider the likelihood of each chronological sequence event by event, "
    prompt += format_story(story, "Sequence") + chain_of_thought
    output, time = runner(prompt, length=64)

    prompt = output + "### Provide your final answer in the format X Y where X is the more unlikely sequence, "
    prompt += "and Y is the event that contradicts with other events in the sequence. ### Answer: "
    output, second_time = runner(prompt, "grammar/answer.gbnf", length=4)
    print(output)

    try:
        response = output.split("### Answer: ")
        reason = response[0].split(chain_of_thought)
        answer = response[1]
        result = get_story_info(story) | {
            "story": answer[0],
            "sentence": int(answer[2]),
            "reason": reason[1],
            "time": time + second_time
        }
    except Exception:
        print(f"@@@ Bad output: {output}")
        return {}

    return result


def reasoning_run(runner, story):
    prompt = "### Which of the following two chronological sequences of events is more physically unlikely? "
    return reasoning_run_impl(runner, story, prompt)


def oneshot_run(runner, story, example):
    example_sentence = get_conflicting(example)
    example_story = example["stories"][1 - example["label"]]

    prompt = f"### Here is an example of a physically unlikely sequence of events; it is unlikely because "
    prompt += f"events {example_sentence[0]} and {example_sentence[1]} contradict each other: "
    prompt += format_sentences(example_story["sentences"])
    prompt += "### Which of the following two chronological sequences of events is more physically unlikely? "
    return reasoning_run_impl(runner, story, prompt)


def run(arguments):
    dataset = load_dataset("sled-umich/TRIP")
    dataset = dataset["OrderTest"]
    dataset = dataset.shuffle(seed=42)

    example = dataset.skip(len(dataset) - 1)[0]
    dataset = dataset.take(arguments.samples)

    match arguments.experiment:
        case "baseline":
            action_llama = lambda item: baseline_run(run_llama, item)
            action_bitnet = lambda item: baseline_run(run_bitnet, item)
        case "reasoning":
            action_llama = lambda item: reasoning_run(run_llama, item)
            action_bitnet = lambda item: reasoning_run(run_bitnet, item)
        case "oneshot":
            action_llama = lambda item: oneshot_run(run_llama, item, example)
            action_bitnet = lambda item: oneshot_run(run_bitnet, item, example)
        case _:
            print(f"Unknown experiment {arguments.experiment}")
            return

    if not arguments.no_llama:
        results = pool_execute(dataset, action_llama)
        save_list(f"{arguments.experiment}-llama.txt", results)

    if not arguments.no_bitnet:
        results = pool_execute(dataset, action_bitnet)
        save_list(f"{arguments.experiment}-bitnet.txt", results)


def print_result(name):
    results = load_list(name)
    if len(results) == 0:
        return

    count_story = list(result["story"] == result["correct_story"] for result in results).count(True)
    count_sentence = list(result["sentence"] in result["correct_sentence"] for result in results).count(True)

    import statistics
    average_time = statistics.fmean(float(result["time"]) for result in results)
    count_total = len(results)

    print(name)
    print(f"Correct story: {count_story} / {count_total} = {count_story / count_total * 100}%")
    print(f"Correct sentence: {count_sentence} / {count_total} = {count_sentence / count_total * 100}%")
    print(f"Average inference time: {average_time}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", "-e", type=str, required=True)
    parser.add_argument("--samples", "-s", type=int, default=100)
    parser.add_argument("--no-run", "-n", action="store_true")
    parser.add_argument("--no-llama", action="store_true")
    parser.add_argument("--no-bitnet", action="store_true")

    arguments = parser.parse_args()
    if not arguments.no_run:
        run(arguments)

    print_result(f"{arguments.experiment}-llama.txt")
    print_result(f"{arguments.experiment}-bitnet.txt")


if __name__ == '__main__':
    main()
