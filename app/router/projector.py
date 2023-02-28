import os
import time
from fastapi import APIRouter, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel
from typing import List, Optional
from models.beautygan.beautygan_model import get_beautygan, transfer
import base64, io


router = APIRouter(
    prefix="/projector",
    tags=["projector"],
)


class TransferImage(BaseModel):
    result: Optional[List[str]]
    # pydantic custom class 허용
    class Config:
        arbitrary_types_allowed = True


@router.post("/")
async def make_transfer(files: List[UploadFile] = [File(...)], model=Depends(get_beautygan)):
    sess, graph = model
    image_bytes = await files[0].read()  # user image
    ref_bytes = await files[1].read()  # refer image

    # np.ndarray -> PIL 이미지 -> ASCII코드로 변환된 bytes 데이터(str)
    try:
        transfer_result, transfer_refer = transfer(sess, graph, image_bytes, ref_bytes)
    except:  # BeautyGAN 적용 불가 이미지
        product = {"result": "Incorrect"}
        return product

    product = TransferImage(result=[transfer_result, transfer_refer])

    return product
