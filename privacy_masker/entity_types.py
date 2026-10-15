"""組み込みentity type定数。entity_typeは自由文字列であり、これは推奨語彙。"""

PERSON = "PERSON"
ORG = "ORG"
LOCATION = "LOCATION"
ADDRESS = "ADDRESS"
EMAIL = "EMAIL"
PHONE_NUMBER = "PHONE_NUMBER"
POSTAL_CODE = "POSTAL_CODE"
CREDIT_CARD = "CREDIT_CARD"
IP_ADDRESS = "IP_ADDRESS"
MY_NUMBER = "MY_NUMBER"
DATE = "DATE"
# v0.8: 日本固有の識別子と汎用型（Azure PII / Google SDP の日本カバレッジに合わせた）
CORPORATE_NUMBER = "CORPORATE_NUMBER"      # 法人番号（13 桁、検査用数字）
DRIVERS_LICENSE = "DRIVERS_LICENSE"        # 運転免許証番号（12 桁、文脈必須）
PASSPORT = "PASSPORT"                      # 旅券番号（英字 2 + 数字 7）
RESIDENCE_CARD = "RESIDENCE_CARD"          # 在留カード番号（英字 2 + 数字 8 + 英字 2）
PENSION_NUMBER = "PENSION_NUMBER"          # 基礎年金番号（4 桁 + 6 桁、文脈必須）
BANK_ACCOUNT = "BANK_ACCOUNT"              # 口座番号（7 桁、文脈必須）
URL = "URL"
MAC_ADDRESS = "MAC_ADDRESS"
AGE = "AGE"
USERNAME = "USERNAME"                      # HandleDetector
