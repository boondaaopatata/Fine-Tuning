from transformers import AutoTokenizer,AutoModelForCausalLM

def Create_Queue():
    class Queue:
        def __init__(self):
            self.queue = []

        def enqueue(self, data):
            self.queue.append(data)

        def dequeue(self):
            if not self.isEmpty():
                return self.queue.pop(0)
            return "Queue Underflow!"
        
        def isEmpty(self):
            return len(self.queue) == 0
        
        def lengthOfQueue(self):
            return len(self.queue)
        
    is_running = True
    while is_running:
        try:
            num_of_prompts = int(input("Enter how many propmts you want to enter: "))
        except ValueError:
            print("Invalid Value.Please enter a number")
        else:
            is_running = False

    Q_list = Queue()
    for round in range(num_of_prompts):
        is_running = True
        while is_running:
            try:
                prompt = input(f"Enter propmt number {round + 1}: ")
            except ValueError:
                print("Invalid Value.Please enter a number")
            else:
                is_running = False
        Q_list.enqueue(prompt)

    return Q_list




tok2 = AutoTokenizer.from_pretrained("./ft")
tok2.pad_token = tok2.eos_token
m2 = AutoModelForCausalLM.from_pretrained("./ft")
m2.config.pad_token_id = m2.config.eos_token_id

num = Create_Queue()

for x in range(num.lengthOfQueue()):
    inputs = tok2(num.dequeue(), return_tensors="pt")
    outputs = m2.generate(
        **inputs,
        max_length=128 + 60,
        do_sample=True,
        top_p=0.9,
        pad_token_id=m2.config.pad_token_id
    )
    print("______________________________________________________________________________")
    print(tok2.decode(outputs[0], skip_special_tokens=True))
