from __future__ import annotations

import os
from pathlib import Path

import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def main() -> None:
    load_dotenv()

    cloud_name = require_env("CLOUDINARY_CLOUD_NAME")
    api_key = require_env("CLOUDINARY_API_KEY")
    api_secret = require_env("CLOUDINARY_API_SECRET")

    image_path = Path(
        "output_images/job_opening_central_bank_of_india_invites_offline_applications_for_flc_counsellor_posts.png"
    )
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )

    result = cloudinary.uploader.upload(
        str(image_path),
        folder="autoflow/generated",
    )

    print("secure_url:", result.get("secure_url"))
    print("public_id :", result.get("public_id"))


if __name__ == "__main__":
    main()
