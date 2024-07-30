import asyncio
import logging
import sys
from dotenv import dotenv_values
import aiohttp

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

configs = dotenv_values(".env")


dp = Dispatcher()


@dp.message(Command("classify"))
async def classify_handler(message: Message) -> None:

    def progress_bar(percent: float) -> str:
        return f"[{int(percent/0.1)*'█'}{(10-int(percent/0.1))*'─'}]"

    if len(message.text.split()) > 1:
        text = " ".join(message.text.split()[1:])
    elif message.reply_to_message:
        text = message.reply_to_message.text
    else:
        await message.answer("Nothing to classify.")
        return
    async with aiohttp.ClientSession() as session:
        async with session.post(configs["CLASSIFIER_URL"], json={"text": text}) as resp:
            if resp.status == 200:
                resp_json = await resp.json()
                await message.reply(
                    f"Y <b>{round(resp_json[0][1], 2)}</b> {progress_bar(resp_json[0][1])} <b>{round(resp_json[0][0], 2)}</b> N"
                    + f"\n\nIs Santilla: <b>{bool(resp_json[1]=='POSITIVE')}</b>"
                )
            else:
                await message.answer("Something went wrong.")


async def main() -> None:
    bot = Bot(
        token=configs["TOKEN"], default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
