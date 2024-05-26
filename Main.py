from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, pipeline, AutoProcessor, SeamlessM4TModel
from parler_tts import ParlerTTSForConditionalGeneration
import cv2
import torch
import soundfile as sf
import sounddevice as sd
import numpy as np

IMAGE_TO_TEXT_MODEL = "microsoft/trocr-large-handwritten"
GENERATIVE_MODEL = "gpt2"
TEXT_TO_SPEECH_MODEL = "facebook/hf-seamless-m4t-medium"
TTS_DESCRIPTION = "A female speaker with a slightly low-pitched voice delivers her words calmly with clear audio."
AUDIO_FILE_PATH = "./speech/parler_tts_out.wav"
TEXT_FILE_PATH = "./text/generated_text.txt"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Image_to_text():
    def __init__(self):
        print("DOWNLOADING MODELS")
        set_seed(42)

        print("\n", "--------------------------Image_to_text model--------------------------")
        self.itt_proc = TrOCRProcessor.from_pretrained(IMAGE_TO_TEXT_MODEL)
        self.itt_model = VisionEncoderDecoderModel.from_pretrained(IMAGE_TO_TEXT_MODEL)

        print("\n", "--------------------------Generative model--------------------------")
        self.gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gen_model =  GPT2LMHeadModel.from_pretrained("gpt2")

        print("\n", "--------------------------Text_to_speech model--------------------------")
        self.tts_processor = AutoProcessor.from_pretrained(TEXT_TO_SPEECH_MODEL)
        self.tts_model = SeamlessM4TModel.from_pretrained(TEXT_TO_SPEECH_MODEL)
        print("MODELS ARE READY")

        print("CAMERA SETUP")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("!!!Can't open camera!!!")
            exit()
        ret, frame = cap.read()
        if not ret:
                print("Can't receive frame.Exiting...")
                exit
        self.cap = cap

        self.top_left = (int(frame.shape[1] * 0.125), int(frame.shape[0] * 0.375))
        self.bottom_right = (int(frame.shape[1] * 0.875), int(frame.shape[0] * 0.625))
        self.color = (0, 255, 0)
        self.color2 = (0, 0, 255)
        self.thickness = 3
        self.font_scale = 1
        self.font_thickness = 1
        self.image_text = "..."
        self.error_text = "..."
        self.generated_text = "..."
        self.is_text_generated = False
        self.is_speech_generated = False

    def convert_to_text(self, frame):
        print("Extracting TEXT from IMAGE")
        frame = frame[int(frame.shape[0] * 0.375) + 3: int(frame.shape[0] * 0.625) - 3,
                      int(frame.shape[1] * 0.125) + 3: int(frame.shape[1] * 0.875) - 3]

        pixel_values = self.itt_proc(images=frame, return_tensors="pt").pixel_values
        generated_ids = self.itt_model.generate(pixel_values, max_length=30)
        generated_text = self.itt_proc.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.image_text = generated_text
        self.is_text_generated = False
        self.is_speech_generated = False
        print(self.image_text)


    def generate_text(self):
        print("Generating extra TEXT from extracted")
        if self.is_text_generated:
            self.show_generated_text()

        if len(self.image_text) < 2 or self.image_text == "...":
            print("NO TEXT FROM IMAGE")
            self.error_text = "No text from image!"
            self.image_text = "..."
        else:
            print(f"Text from image: {self.image_text}")
            input_ids = self.gen_tokenizer.encode(self.image_text, return_tensors="pt")
            output = self.gen_model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
            )

            text = self.gen_tokenizer.decode(output[0], skip_special_tokens=True)
            self.generated_text = text
            self.show_generated_text()
            f = open(TEXT_FILE_PATH, "w")
            f.write(text)
            f.close()
            self.is_text_generated = True
            self.is_speech_generated = False

    def show_generated_text(self):
        print(self.generated_text)
        cv2.namedWindow("Generated text")
        img = 255 * np.ones((200, 800, 3), dtype=np.uint8)
        text = self.generated_text.replace('\n\n', '\n').replace("\n", " ")
        start = 0
        N = 10
        lines = text.split()
        y = 30
        for stop in range(N, len(lines) + N, N):
            cv2.putText(img,
                        " ".join(lines[start:stop]),
                        (10,y),
                        cv2.FONT_HERSHEY_PLAIN,
                        self.font_scale,
                        (0,0,0),
                        self.font_thickness)
            start = stop
            y += 20
        cv2.imshow("Generated text", img)

    def read_generated_text(self):
        if self.is_speech_generated:
            self.play_audio()
            return

        if len(self.generated_text) < 2 or self.generated_text == "...":
            print("NO GENERATED TEXT")
            self.error_text = "No generated text!"
            self.generated_text = "..."
        else:
            print("Generating AUDIO from TEXT")
            text_inputs = self.tts_processor(text = self.generated_text, src_lang="eng", return_tensors="pt")
            audio_arr = self.tts_model.generate(**text_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
            sf.write(AUDIO_FILE_PATH, audio_arr, self.tts_model.config.sampling_rate)
            self.is_speech_generated = True
            self.play_audio()

    def play_audio(self):
        print("PLAY AUDIO")
        data, samplerate = sf.read(AUDIO_FILE_PATH)
        sd.play(data, samplerate)

    def add_overlay(self, frame):
        cv2.rectangle(frame, self.top_left, self.bottom_right, self.color, self.thickness)
        cv2.putText(frame, 'C - capture frame   G - generate text   S - text to speech',
                    (10,30),
                    cv2.FONT_HERSHEY_PLAIN,
                    self.font_scale,
                    self.color,
                    self.font_thickness)
        cv2.putText(frame, 'Q - quit',
                    (10,60),
                    cv2.FONT_HERSHEY_PLAIN,
                    self.font_scale,
                    self.color2,
                    self.font_thickness)
        cv2.putText(frame, f'Text from image: {self.image_text}',
                    (10 ,frame.shape[0] -60),
                    cv2.FONT_HERSHEY_PLAIN,
                    self.font_scale,
                    self.color,
                    self.font_thickness)
        cv2.putText(frame, f'Error: {self.error_text}',
                    (10 ,frame.shape[0] -30),
                    cv2.FONT_HERSHEY_PLAIN,
                    self.font_scale,
                    self.color2,
                    self.font_thickness)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame.Exiting...")
                exit

            self.add_overlay(frame)

            cv2.imshow('Image_to_audio', frame)

            key = cv2.waitKey(1)
            match key:
                case 99: # C
                    print("CONVERT IMAGE TO TEXT")
                    self.convert_to_text(frame)
                case 103: # G
                    print("GENERATE TEXT")
                    self.generate_text()
                case 115: # S
                    print("TEXT TO SPEECH")
                    self.read_generated_text()
                case 113: # Q
                    break

        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    image_to_text = Image_to_text()
    image_to_text.run()

if __name__ == '__main__':
    main()
