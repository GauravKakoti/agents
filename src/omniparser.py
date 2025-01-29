# Imports from omniparser.py
import base64
import io
# utility function
import os
import subprocess
# imports from utils.py# from ultralytics import YOLO
import time
# additional imports
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import cv2
import easyocr
import numpy as np
import supervision as sv
import torch
import torchvision.transforms as T
# %matplotlib inline
from matplotlib import pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision.ops import box_convert
from torchvision.transforms import ToPILImage
from playwright.async_api import Page
import asyncio
import re
# used for caption model
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from ultralytics import YOLO
# For Ollama-based caption model
from ollama import Client as OllamaClient


@dataclass
class OmniParserConfig:
    som_model_path: str = "src/weights/omniparser/icon_detect/best.pt"
    caption_model: str = "blip2"  # alternate choice: llava
    caption_model_path: str = "src/weights/omniparser/icon_caption_blip2"
    device: str = "cpu"

    # detection thresholds
    conf_threshold: float = 0.05  # From the original code base
    text_threshold: float = 0.05
    iou_threshold: float = 0.05

    has_draw_box_config: bool = True
    text_scale: float = 0.8
    text_thickness: int = 2
    text_padding: int = 3
    thickness: int = 3


class OCRReader(easyocr.Reader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _perform_ocr(self, image_input: Union[str, bytes], **easyocr_args):
        """
        Private method to perform OCR on the inpute image and return detected texts and bounding boxes.
        Easy OCR is built to that image can be a path to the image, the image as bytes, or the numpy array

        Args:
            image_path (str or bytes or np.array): path to the image file, bytes of the image, or the numpy array of the image
            text_threshold (float, optional): Text detection threshold for OCR
            ***easyocr_args: Additional arguments to pass to easyocr

        Returns:
            Tuple of lists:
                - List of recognized text strings
                - List of Bounding box coordinates for each detected text.
        """

        results = self.readtext(image_input, **easyocr_args)
        texts = [item[1] for item in results]
        bboxes = [item[0] for item in results]
        return texts, bboxes

    def _convert_bboxes(self, bboxes, output_format: str = "xywh"):
        converted_boxes = []
        for bbox in bboxes:
            if output_format == "xywh":
                x, y, w, h = (
                    bbox[0][0],
                    bbox[0][1],
                    bbox[2][0] - bbox[0][0],
                    bbox[2][1] - bbox[0][1],
                )
                x, y, w, h = int(x), int(y), int(w), int(h)
                converted_box = (x, y, w, h)
            elif output_format == "xyxy":
                x, y, xp, yp = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
                x, y, xp, yp = int(x), int(y), int(xp), int(yp)
                converted_box = (x, y, xp, yp)
            else:
                raise ValueError("Invalid output format. Choose 'xywh' or 'xyxy'")
            converted_boxes.append(converted_box)

        return converted_boxes

    def _display_image_with_bboxes(
        self,
        image_input: Union[str, bytes],
        bboxes: List[Tuple[int, int, int, int]],
        texts: List[str],
        bbox_format: str,
    ):
        """
        Private method to display the image with bounding boxes and labels.

        Args:
            image_path (str): Path to the image file.
            bboxes (List[Tuple[int, int, int, int]]): List of bounding box coordinates.
            texts (List[str]): List of text labels.
            bbox_format (str): Format of the bounding boxes ('xywh' or 'xyxy').
        """
        if isinstance(image_input, bytes):
            # convert bytes to a numpy array
            np_arr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(np_arr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, str):
            # read image from path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("image_input should be a byte string or an image path")

        # Draw bounding boxes and labels
        for bbox, text in zip(bboxes, texts):
            if bbox_format == "xywh":
                x, y, w, h = bbox
                x2, y2 = x + w, y + h
            elif bbox_format == "xyxy":
                x, y, x2, y2 = bbox

            # Draw rectangle
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
            # Put label
            # cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the image
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def perform_ocr(
        self,
        image_input: Union[str, any],
        display_img: bool = False,
        output_bb_format: str = "xywh",
        **easyocr_args,
    ):
        """
        Performs OCR on the image and returns the recognized texts and their bounding boxes in the desired format.

        Args:
            image_path (str): path to image file.
            display_image (bool, optional): Whether to display the image with bounding boxes, defaults to False
            output_bb_format (str, optional): Format of the output bounding boxes ('xywh' or 'xyxy')
            **easyocr_args: Additional arguments to pass to easyocr

        Returns:
            Tuple[List[str], List[Tuple[int, int, int, int]]]:
                - List of recognized text strings
                - List of bounding box coordinates in the specified format
        """

        texts, bboxes = self._perform_ocr(image_input=image_input, **easyocr_args)
        bboxes = self._convert_bboxes(bboxes, output_format=output_bb_format)

        if display_img:
            self._display_image_with_bboxes(
                image_input=image_input,
                bboxes=bboxes,
                texts=texts,
                bbox_format=output_bb_format,
            )

        return texts, bboxes


class SOMModel(YOLO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform_icon_detection(
        self, image_input: Union[str, bytes], conf_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Method to perform Icon detection on an image.
        Args:
            image_input: path to image or bytes
        Output:
            detections (List): a list of detection dictionaries
        """
        if isinstance(image_input, str):
            # This is a path to an image, which can just be passed to the YOLO model.
            image = image_input
        elif isinstance(image_input, bytes):
            # We must convert the byte string into a PIL image
            image = Image.open(io.BytesIO(image_input))
        else:
            raise ValueError("image_input must be a string or bytes.")

        results = self.predict(source=image, conf=conf_threshold)
        detections = []

        for result in results:
            boxes_xyxy = result.boxes.xyxy
            boxes_xyxyn = result.boxes.xyxyn
            boxes_xywh = result.boxes.xywh
            boxes_xywhn = result.boxes.xywhn
            confidences = result.boxes.conf
            class_ids = result.boxes.cls
            class_names = [
                self.names[int(cls_id)] for cls_id in class_ids
            ]  # Class names

            for (
                bbox_xyxy,
                bbox_xyxyn,
                bbox_xywh,
                bbox_xywhn,
                confidence,
                class_id,
                class_name,
            ) in zip(
                boxes_xyxy,
                boxes_xyxyn,
                boxes_xywh,
                boxes_xywhn,
                confidences,
                class_ids,
                class_names,
            ):
                detection = {
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_xyxyn": bbox_xyxyn,
                    "bbox_xywh": bbox_xywh,
                    "bbox_xywhn": bbox_xywhn,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                }

                detections.append(detection)

        return detections


class CaptionModelProcessor:
    def __init__(
        self,
        caption_model_path: str = "src/weights/omniparser/icon_caption_blip2",
        device: str = "cpu",
    ):
        self.device = device
        self.processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )  # Currently hard coded, not sure why this one specifically...
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            caption_model_path,
            device_map=None,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        )
        self.model.to(self.device)

    def caption_icons(self, cropped_images: List[Image.Image], prompt: str = None):
        batch_size = 10  # TODO(dominic): We should make this more dynamic?

        if not prompt:  # This was taken directly from the original
            prompt = "The image shows"
            # TODO(dominic): Commented the following since its not used, but we may want to retain it for later?
            # if 'florence' in model.config.name_or_path:
            #     prompt = "<CAPTION>"
            # else:
            #     prompt = "The image shows"

        device = self.model.device
        generated_texts = []
        for idx in range(0, len(cropped_images), batch_size):
            batch = cropped_images[idx : idx + batch_size]
            if self.model.device.type != "cpu":
                inputs = self.processor(
                    images=batch, text=[prompt] * len(batch), return_tensors="pt"
                ).to(device=device, dtype=torch.float16)
            else:
                inputs = self.processor(
                    images=batch, text=[prompt] * len(batch), return_tensors="pt"
                ).to(device=device)

            # run inputs through the model
            generated_ids = self.model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_return_sequences=1,
            )  # temperature=0.01, do_sample=True,
            # if 'florence' in model.config.name_or_path:
            #     generated_ids = self.model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=1024,num_beams=3, do_sample=False)
            # else:
            #     generated_ids = self.model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1) # temperature=0.01, do_sample=True,

            # decode the generated_ids and add the to generated_texts
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            generated_text = [gen.strip() for gen in generated_text]
            generated_texts.extend(generated_text)

        return generated_texts


class OllamaCaptionModel(CaptionModelProcessor):
    def __init__(self, model="llava:7b"):
        self.client = OllamaClient()
        self.prompt = "The image is a small part of a web page screenshot. Assume the image\
 is not blurry, and do not mention that this is a screen or webpage or that it is on a computer.\
 As concisely as possible, describe what the image shows and how a user might interact with it.\
 Use fewer than 10 words"
        if str(subprocess.check_output("ollama list", shell=True)).find(model) < 0:
            print("model not detected, downloading")
            subprocess.run("ollama pull " + model, shell=True)
        self.model = model

    
    def caption_icons(self, cropped_images: List[Image.Image], prompt: str = None, padding=15):
        """ Caption each image in cropped_images using the prompt. Padding is added as a white border around the image """
        if prompt is None:
            prompt = self.prompt
        responses = []
        for image in cropped_images:
            imgByteArr = io.BytesIO()
            # if type(image) == str:
            #     image = Image.open(image)
            ImageOps.expand(image, border=padding, fill="white").save(imgByteArr, format="png")  # image.format)
            image = imgByteArr.getvalue()
            message = {'role': 'user', 
                    'content': prompt,
                    'images': [image]}
            response = self.client.chat(model=self.model, messages=[message])
            responses += [response["message"]["content"].strip()]
        return responses



class OmniParser:
    def __init__(
        self,
        som_model: SOMModel,
        caption_model_processor: CaptionModelProcessor,
        ocr_reader: OCRReader,
    ):
        self.som_model = som_model
        self.caption_model_processor = caption_model_processor
        self.ocr_reader = ocr_reader

    @classmethod
    def from_config(cls, config: OmniParserConfig) -> "OmniParser":
        """
        Class method to initialize an OmniParser instance using an OmniParserConfig instance.

        Args:
            config (OmniParserConfig): Configuration object containing paths, device info, and thresholds.

        Returns:
            OmniParser: A new instance of OmniParser initialized with the specified configuration.
        """
        # Initialize the OCRReader, SOMModel, and CaptionModelProcessor based on the config
        ocr_reader = OCRReader(["en"], gpu=config.device != "cpu")
        som_model = SOMModel(config.som_model_path)
        if config.caption_model == "blip2":
            caption_model_processor = CaptionModelProcessor(
                config.caption_model_path, device=config.device
            )
        elif config.caption_model == "llava":
            caption_model_processor = OllamaCaptionModel()
        else:
            raise Exception("I cannot yet handle the provide model: " + str(config.caption_model))

        # Return the new instance
        return cls(
            som_model=som_model,
            caption_model_processor=caption_model_processor,
            ocr_reader=ocr_reader,
        )

    def _remove_overlaps(self, boxes, iou_threshold, ocr_bboxes=None):
        '''
        Handles overlaps among the boxes and ocr_bboxes. If two overlap sufficiently, it
        removes the larger one, so we focus on more details
        Args:
            boxes: a list of boxes around potential icons
            iou_threshold: what fraction of a box should be covered by another before removing
            ocr_bboxes: a list of boxes around text
        returns:
            torch.Tensor(n, 4) tensor with each box's dimensions (left, top, right, bottom)
        '''
        assert ocr_bboxes is None or isinstance(ocr_bboxes, List)
        if not ocr_bboxes:
            ocr_bboxes = []

        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        def intersection_area(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            return max(0, x2 - x1) * max(0, y2 - y1)

        def IoU(box1, box2):
            intersection = intersection_area(box1, box2)
            union = box_area(box1) + box_area(box2) - intersection + 1e-6
            if box_area(box1) > 0 and box_area(box2) > 0:
                ratio1 = intersection / box_area(box1)
                ratio2 = intersection / box_area(box2)
            else:
                ratio1, ratio2 = 0, 0
            return max(intersection / union, ratio1, ratio2)

        boxes = boxes.tolist()
        filtered_boxes = []
        # Assume the ocr_bboxes are good
        filtered_boxes.extend(ocr_bboxes)
        # Remove icon bounding boxes which have significant overlap with smaller icon bounding boxes or which overlap significantly with a ocr bbox.
        for i, box1 in enumerate(boxes):
            is_valid_box = True
            for j, box2 in enumerate(boxes):
                if (
                    i != j
                    and IoU(box1, box2) > iou_threshold
                    and box_area(box1) > box_area(box2)
                ):
                    is_valid_box = False
                    break
            if is_valid_box:
                # if it overlaps with text, the text wins
                if ocr_bboxes:
                    if not any(
                        IoU(box1, box3) > iou_threshold
                        for box3 in ocr_bboxes
                    ):
                        filtered_boxes.append(box1)
        return torch.tensor(filtered_boxes)

    def _crop_detected_icons(
        self, image: Image.Image, filtered_boxes, ocr_bboxes=None, padding=3
    ) -> List[Image.Image]:
        """
        Crops detected icons from the image using provided bounding boxes.

        Args:
            image (Image.Image): The source image.
            filtered_boxes (Tensor): Bounding boxes in normalized coordinates (x_min, y_min, x_max, y_max).
            ocr_bboxes (Optional[Tensor]): OCR bounding boxes to exclude from cropping, if provided.
            padding (int): how many extra pixels to include around the boundary of icons

        Returns:
            List[Image.Image]: List of cropped icon images.
        """
        # Calculate the list of icon boxes by excluding OCR boxes if they exist
        non_ocr_boxes = (
            filtered_boxes[len(ocr_bboxes) :] if ocr_bboxes else filtered_boxes
        )

        # Calculate dimensions of the original image once
        img_width, img_height = image.size

        # Crop each box and convert it to PIL format
        cropped_images = []
        for coord in non_ocr_boxes:
            x_min, y_min, x_max, y_max = (
                coord[0] * img_width - padding,
                coord[1] * img_height - padding,
                coord[2] * img_width + padding,
                coord[3] * img_height + padding,
            )

            # Crop the image using the PIL `crop` method with integer bounding box values
            cropped_image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            cropped_images.append(cropped_image)

        return cropped_images

    def annotate(
        self, image_source: np.ndarray,
        raw_bboxes: list[dict],
        phrases: List[str] = None,
        output_coord_in_ratio=False,
        **kwargs
    ) -> np.ndarray:
        """
        This function annotates an image with bounding boxes and labels.

        Parameters:
        image_source (bytes or string): The source image to be annotated.
        raw_bboxes (List[dict]): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
        phrases (List[str]): A list of labels for each bounding box.
        output_coord_in_ratio (bool): if true, assumes input is as a fraction of the screen

        Returns:
        base64: Encoded version of the annotated image
        """
        # get the associated PIL image object, must convert to RGB for annotation purposes
        # get dimensions of the image
        if isinstance(image_source, str):
            image = np.asarray(Image.open(image_source).convert("RGB"))
        elif isinstance(image_source, bytes):
            image = np.asarray(Image.open(io.BytesIO(image_source)).convert("RGB"))
        else:
            raise ValueError("image_source must be a string or bytes.")
        
        # if there are no labels, just label numerically
        if phrases is None:
            phrases = range(len(raw_bboxes))
        labels = [str(phrase) for phrase in phrases]

        xyxy = np.array([
            [
                bbox["shape"]["x"],
                bbox["shape"]["y"],
                bbox["shape"]["x"] + bbox["shape"]["width"],
                bbox["shape"]["y"] + bbox["shape"]["height"]]
            for bbox in raw_bboxes
        ])
        if output_coord_in_ratio:
            h, w, _ = image.shape
            xyxy = raw_bboxes * np.array([w, h, w, h])[None, :]
        detections = sv.Detections(xyxy=xyxy, class_id=np.arange(xyxy.shape[0]))

        # from util.box_annotator import BoxAnnotator
        # box_annotator = BoxAnnotator(color=sv.ColorPalette.DEFAULT, text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
        box_annotator = sv.BoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK, text_padding=3, text_scale=0.4, text_thickness=1
        )
        annotated_frame = image.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # Encode the image
        pil_image = Image.fromarray(annotated_frame)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded_image

    def get_bboxes(
        self,
        image_input: Union[str, bytes],
        iou_threshold: float = 0.8,
        conf_threshold: float = 0.5,
        use_local_semantics: bool = False,
        output_coord_in_ratio=False,
        icon_padding=3,
        **kwargs
    ):
        """
        Go through the image, mark boxes around text and icons, and generate descriptions
        for each box including its location, and what the text said or the icon indicates
        Parameters:
        image_input: the image to be parsed
        iou_threshold (float): how much of a box should overlap with another before the subsumed box is removed
        conf_threshold (float): How confident should we be before adding an icon to the list
        use_local_semantics (bool) whether or not to run the captioning model on non-text boxes
        output_coord_in_ratio (bool): rather than pixel coordinates, give box coordinates in a fraction of the screen
        icon_padding (int): number of extra pixels around the border to include when processing the captions

        Returns:
        List[dict]: bbox data, namely x, y, type, text, and arialabel
        """

        # OCR the image
        text, ocr_bboxes = self.ocr_reader.perform_ocr(
            image_input, output_bb_format="xyxy"
        )

        # get the associated PIL image object, must convert to RGB for annotation purposes
        # get dimensions of the image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            raise ValueError("image_input must be a string or bytes.")

        w, h = image.size

        # Normalize the OCR bounding boxes
        if ocr_bboxes:
            ocr_bboxes_normed = [
                (x_1 / w, y_1 / h, x_2 / w, y_2 / h)
                for (x_1, y_1, x_2, y_2) in ocr_bboxes
            ]
        else:
            ocr_bboxes_normed = []

        # Perform the icon detection
        icon_detections = self.som_model.perform_icon_detection(
            image_input=image_input, conf_threshold=conf_threshold
        )  # TODO(dominic): deal with passing conf_threshold

        # Check if there are any detections before proceeding
        if not icon_detections:  # Handle the case where no icons are detected
            icon_bboxes_normed = torch.empty((0, 4)) #Empty Tensor
        else:
            icon_bboxes_normed = torch.cat(
                [detection["bbox_xyxyn"].unsqueeze(0) for detection in icon_detections],
                axis=0,
        )  # create a matrix of icon detections

        # remove the overlapping boxes amongst the ocr bboxes and icon bboxes, keeps all ocr
        filtered_boxes = self._remove_overlaps(
            boxes=icon_bboxes_normed,
            iou_threshold=iou_threshold,
            ocr_bboxes=ocr_bboxes_normed,
        )

        # Caption the icons and text
        # First the text, that's easy
        box_text = [
            f"Text Box ID {i}: {txt}" for i, txt in enumerate(text)
        ]
        # Now the icons
        if use_local_semantics:
            cropped_images = self._crop_detected_icons(
                image, filtered_boxes=filtered_boxes, ocr_bboxes=ocr_bboxes, padding=icon_padding
            )
            captions = self.caption_model_processor.caption_icons(
                cropped_images=cropped_images
            )  # TODO(dominic): We should experiment with passing in prompts
        else:
            # If we don't run the captioner, just put "None" for all
            num_ocr_bboxes = len(ocr_bboxes) if ocr_bboxes else 0
            num_icons = len(filtered_boxes) - num_ocr_bboxes
            captions = ["None"] * num_icons
        icon_start = len(box_text)
        icon_text = [
            f"Icon Box ID {i + icon_start}: {txt}" for i, txt in enumerate(captions)
        ]
        box_text.extend(icon_text)

        try:
            xywh = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="xywh")
            if not output_coord_in_ratio:
                xywh *=  torch.Tensor([w, h, w, h])
        except ValueError:
            return []


        # formating output
        bbox_list = [
            {
                "from": "omniparser",
                "shape": {
                    "x": coord[0].item(),
                    "y": coord[1].item(),
                    "width": coord[2].item(),
                    "height": coord[3].item(),
                },
                "text": text.split(": ", 1)[1],
                "type": text.split(" ", 1)[0].lower(),
            }
            for text, coord in zip(box_text, xywh)
        ]

        return bbox_list

    def parse(
        self,
        image_input: Union[str, bytes],
        iou_threshold: float = 0.8,
        conf_threshold: float = 0.5,
        use_local_semantics: bool = False,
        output_coord_in_ratio=False,
        icon_padding=3,
        phrases=None,
        **kwargs
    ):
        """
        Go through the image, generate descriptions for each piece of text or icon,
        including its location, size of a boudning box, and what the text said or the icon indicates
        Parameters:
        image_input: the image to be parsed
        iou_threshold (float): how much of a box should overlap with another before the subsumed box is removed
        conf_threshold (float): How confident should we be before adding an icon to the list
        use_local_semantics (bool) whether or not to run the captioning model on non-text boxes
        output_coord_in_ratio (bool): rather than pixel coordinates, give box coordinates in a fraction of the screen
        icon_padding (int): number of extra pixels around the border to include when processing the captions
        phrases (list): list of labels to give each of the bounding boxes, defaults to numbering them 
        
        Returns:
        base64: Encoded version of the annotated image
        List[dict]: bbox data, namely x, y, type, text, and arialabel
        """
        bboxes = self.get_bboxes(image_input, iou_threshold, conf_threshold,
            use_local_semantics, output_coord_in_ratio, icon_padding, **kwargs)
        encoded_img = self.annotate(image_input, bboxes, phrases, output_coord_in_ratio, **kwargs)
        return encoded_img, bboxes