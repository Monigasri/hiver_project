from __future__ import annotations
from pathlib import Path
import yaml

DEFAULT_TAXONOMY={"intents":[
 {"name":"account_access","definition":"Cannot sign in, access, or recover an account.","inclusion":["login","password","access"],"exclusion":["payment issues"],"examples":[]},
 {"name":"payment_or_charge","definition":"Charges, payments, billing, or transaction problems.","inclusion":["charged","payment","billing"],"exclusion":["delivery tracking"],"examples":[]},
 {"name":"service_or_product_issue","definition":"A product or service is not working as expected.","inclusion":["not working","error","broken"],"exclusion":["account login"],"examples":[]},
 {"name":"delivery_or_order_status","definition":"Order, delivery, shipping, or tracking status.","inclusion":["order","delivery","tracking"],"exclusion":["billing"],"examples":[]},
 {"name":"information_request","definition":"General question about available service or product information.","inclusion":["how","where","what"],"exclusion":["active failure"],"examples":[]},
 {"name":"complaint_or_feedback","definition":"Feedback or complaint without a more specific operational issue.","inclusion":["disappointed","feedback","poor service"],"exclusion":["specific payment problem"],"examples":[]},
 {"name":"cancellation_or_change","definition":"Request to cancel, change, or modify an existing service/order.","inclusion":["cancel","change","modify"],"exclusion":["new information request"],"examples":[]},
 {"name":"other","definition":"Message does not fit the defined taxonomy.","inclusion":["unclear"],"exclusion":[],"examples":[]}
]}
def save_taxonomy(path: Path, taxonomy: dict=DEFAULT_TAXONOMY) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(taxonomy,sort_keys=False),encoding="utf-8")
def load_taxonomy(path: Path) -> dict:
    data=yaml.safe_load(path.read_text(encoding="utf-8"))
    names=[x["name"] for x in data.get("intents",[])]
    if not names or len(names)!=len(set(names)): raise ValueError("Taxonomy needs unique intent names")
    return data
