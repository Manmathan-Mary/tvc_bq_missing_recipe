import argparse
import yaml
import json

from google.cloud.storage import bucket
from sqlalchemy import create_engine, text
import polars as pl
import warnings

from google.cloud import bigquery
from datetime import timezone, datetime
from logging_config import setup_logger
from google.cloud import storage


warnings.filterwarnings(
    "ignore",
    message="Your application has authenticated using end user credentials"
)

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

ENVIRONMENT = {
    "stg" : {
        "project_id" : "tvc-stg"
    },
    "prod": {
        "project_id" : "tvc-prod-core"
    }
}

logger = setup_logger()
RECIPE_DATABASE = "recipe"
execution_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_recipe_data(engine, params):
    sql = """
        SELECT
          *
        FROM
          denormalized_recipe
        WHERE date_created > '2026-08-09T00:00:00Z'
        order by date_created desc
   """

    return execute_query_return_dataframe(sql, engine, params=params, schema={})

def get_recipe_data_spanner(engine, lb):
   params = {"lower_bound_timestamp": lb}
   sql = """
        select  r.business_unit_id,  
                d.recipe_id, 
                d.denormalized_recipe_id,
                d.recipe_payload ,
                r.date_updated AS event_time, 
                r.product_attribute_filter_payload ,
                '' as event_id  from recipe r , 
        denormalized_recipe d 
        where r.recipe_id=d.recipe_id
        and r.date_updated > :lower_bound_timestamp 
        order by r.date_updated desc
   """
   return execute_query_return_dataframe(sql, engine, params=params, schema={})


def get_recipe_bq_data(project_id: str):
    GET_RECIPE_BQ = f"""
    SELECT * 
        FROM 
    {project_id}.tvc_item_label.recipe
    order by date_updated desc 
    """

    return GET_RECIPE_BQ


def load_config(env):
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f);
    return config[env]

def get_engine(project_id, instance_name, database_name):
    return create_engine(
        f"spanner+spanner:///projects/{project_id}/instances/{instance_name}/databases/{database_name}"
    )

def execute_query_return_dataframe(query: str, engine, params: dict, schema:dict = {}):
    with engine.connect().execution_options(read_only=True) as connection:
        result = connection.execute(text(query), parameters=params)
        if schema:
            return pl.DataFrame(result.fetchall(), schema=schema)
        else:
            return pl.DataFrame(result.fetchall(), schema=result.keys())


def batch_execute_query_return_dataframe(
        query: str,
        engine,
        params: dict,
        schema: dict = None,
        batch_size: int = 100_000
):

    dfs = []
    total_rows = 0
    batch_number = 0

    logger.debug("Starting query execution")

    with engine.connect().execution_options(
            read_only=True,
            stream_results=True
    ) as connection:

        result = connection.execute(text(query), parameters=params)

        columns = list(result.keys())

        logger.debug("Query execution started, fetching rows in batches")

        while True:
            rows = result.fetchmany(batch_size)

            if not rows:
                break

            batch_number += 1
            batch_row_count = len(rows)
            total_rows += batch_row_count

            logger.debug(
                "Processing batch %s | batch_rows=%s | total_rows=%s",
                batch_number,
                batch_row_count,
                total_rows
            )

            batch_df = pl.DataFrame(
                rows,
                schema=schema if schema else columns
            )

            dfs.append(batch_df)

        logger.debug(
            "Finished fetching all batches | total_batches=%s | total_rows=%s",
            batch_number,
            total_rows
        )

    logger.debug("Concatenating %s batch dataframes", len(dfs))

    if dfs:
        final_df = pl.concat(dfs, rechunk=True)
        logger.debug(
            "Final dataframe created | rows=%s | columns=%s",
            final_df.height,
            final_df.width
        )
    else:
        logger.warning("No rows return from query batches")
        final_df = pl.DataFrame(schema=schema)

    return final_df

def get_all_active_sites_sql(project_id: str) -> str:
    GET_ALL_ACTIVE_SITES = f"""
        SELECT *
            FROM `{project_id}.tvc_item_label.recipe`
        ORDER BY business_unit_id, date_created DESC
    """
    return GET_ALL_ACTIVE_SITES


def parse_utc_timestamp(value):
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

        if dt.tzinfo is None:
            raise ValueError("Timestamp must contain a timezone")

        return dt.astimezone(timezone.utc)

    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid UTC timestamp: {value}. "
            f"Expected format: 2026-07-08T00:00:00Z"
        ) from e

def read_bq_to_polars(
        query: str,
        job_config: bigquery.job.QueryJobConfig,
        dry_run: bool = False,
) -> pl.DataFrame | int | pl.Series:
    client = bigquery.Client()

    if dry_run:
        dry_run_config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
        )

        # Preserve any settings from the supplied job_config
        if job_config:
            dry_run_config.query_parameters = job_config.query_parameters
            dry_run_config.default_dataset = job_config.default_dataset
        query_job = client.query(query, job_config=dry_run_config)

        bytes_processed = query_job.total_bytes_processed
        print(f"Estimated bytes processed: {bytes_processed:,}")
        print(f"Estimated MB processed: {bytes_processed / (1024 ** 2):.2f}")
        print(f"Estimated GB processed: {bytes_processed / (1024 ** 3):.2f}")

        return bytes_processed

    arrow_table = (
        client.query(query, job_config=job_config)
        .to_arrow(create_bqstorage_client=True)
    )
    return pl.from_arrow(arrow_table)

def parse_args(parser):

    parser.add_argument(
        "-e", "--env",
        choices=["stg", "prod"],
        required=True,
        help="Target environment.",
    )

    parser.add_argument(
        "-lb", "--lower-bound-timestamp",
        type=parse_utc_timestamp,
        required=True,
        help="Lower bound timestamp in ISO-8601 format, e.g. 2026-07-08T00:00:00Z"
    )

    parser.add_argument(
        "-b", "--bucket-name",
        type=str,
        required=False,
        help="Specify bucket name, if the identified discrepancy needs to be inserted into BQ using GCS file"
    )

    return  parser.parse_args()

from datetime import datetime, timezone


def format_bq_timestamp(value):
    """
    Convert ISO timestamp:
        2026-08-18T10:02:26.079250Z
    to:
        2026-08-18 10:02:26.079250 UTC
    """
    if not value:
        return None

    return value.replace("T", " ").replace("Z", " UTC")


def map_item_labels(item_labels):
        return [
            {
                "itemlabelid": label.get("itemLabelId"),
                "description": label.get("description"),
                "enabled": str(label.get("enabled")).lower(),
                "itemlabelname": label.get("itemLabelName")
            }
            for label in item_labels
        ]


def map_filter(recipe_filter):
    if not recipe_filter:
        return None

    return  {
        "type": recipe_filter.get("type"),
        "itemlabels": map_item_labels(
            recipe_filter.get("itemLabels", [])
        )
    }


def map_product_attribute_filter(product_filter):
    if not product_filter:
        return None

    return {
        "type": product_filter.get("type"),
        "attributes": [
            {
                "languagecode": attribute.get("languageCode"),
                "product_attribute_id": attribute.get("productAttributeId"),
                "values": attribute.get("values", [])
            }
            for attribute in product_filter.get("attributes", [])
        ]
    }



def map_post_application(post_application):
    if not post_application:
        return None

    return {
        "additemlabels": map_item_labels(
            post_application.get("addItemLabels", [])
        ),
        "removeitemlabels": map_item_labels(
            post_application.get("removeItemLabels", [])
        )
    }

def map_site(sites):
    if not sites:
        return []

    return [
        site.get("siteId")
        for site in sites
    ]

def map_recipe(recipe, event_id=" "):
    try:
        denormalized_recipe_id = recipe.get("denormalizedRecipeId")

        mapped_product_attributes = map_product_attribute_filter(
            recipe.get("productAttributeFilter")
        )

        mapped_post_application = map_post_application(
            recipe.get("postApplication")
        )

        mapped_filter = map_filter(
            recipe.get("filter")
        )

        current_timestamp = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")

        return True, {
            "business_unit_id": recipe.get("businessUnitId"),
            "tenant_id": None,
            "recipe_id": recipe.get("recipeId"),
            "type": recipe.get("type"),
            "workflow": recipe.get("workflow"),
            "name": recipe.get("name"),
            "denormalized_recipe_id": denormalized_recipe_id,
            "description": recipe.get("description"),
            "mandatory": str(recipe.get("mandatory")).lower(),
            "sites": map_site(recipe.get("sites")),

            "event_time": recipe.get("dateUpdated"),
            "event_id": event_id,

            "user_created": recipe.get("userCreated"),
            "user_updated": recipe.get("userUpdated"),

            "date_created": recipe.get("dateCreated"),
            "date_updated": recipe.get("dateUpdated"),

            "etl_date_created": current_timestamp,
            "etl_date_updated": current_timestamp,

            "is_deleted": "false",

            "productattributefilter": mapped_product_attributes,
            "filter": mapped_filter,
            "postapplication": mapped_post_application,
        }

    except Exception as ex:
        logger.warning(
            f"Ignoring recipe {recipe.get("denormalizedRecipeId")} due to improper payload: {ex}"
        )
        return False, None

def load_nested_json(project_id, from_local_file=True, file_path=None, bucket_name=None):
    client = bigquery.Client(project=project_id)

    dataset_id = "tvc_item_label"
    table_id = "recipe"

    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    job_config = bigquery.LoadJobConfig()

    job_config.schema = [
        bigquery.SchemaField('business_unit_id', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('tenant_id', 'STRING', mode='NULLABLE'),
        bigquery.SchemaField('recipe_id', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('type', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('workflow', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('name', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('denormalized_recipe_id', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('description', 'STRING', mode='NULLABLE'),
        bigquery.SchemaField('mandatory', 'BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField('sites', 'STRING', mode='REPEATED'),

        bigquery.SchemaField('event_time', 'TIMESTAMP', mode='REQUIRED'),

        bigquery.SchemaField(
            'productattributefilter',
            'RECORD',
            mode='NULLABLE',
            fields=[
                bigquery.SchemaField('type', 'STRING', mode='REQUIRED'),
                bigquery.SchemaField(
                    'attributes',
                    'RECORD',
                    mode='REPEATED',
                    fields=[
                        bigquery.SchemaField(
                            'languagecode', 'STRING', mode='REQUIRED'
                        ),
                        bigquery.SchemaField(
                            'product_attribute_id', 'STRING', mode='REQUIRED'
                        ),
                        bigquery.SchemaField(
                            'values', 'STRING', mode='REPEATED'
                        ),
                    ],
                ),
            ],
        ),

        bigquery.SchemaField(
            'filter',
            'RECORD',
            mode='NULLABLE',
            fields=[
                bigquery.SchemaField('type', 'STRING', mode='REQUIRED'),
                bigquery.SchemaField(
                    'itemlabels',
                    'RECORD',
                    mode='REPEATED',
                    fields=[
                        bigquery.SchemaField(
                            'itemlabelid', 'STRING', mode='REQUIRED'
                        ),
                        bigquery.SchemaField(
                            'description', 'STRING', mode='NULLABLE'
                        ),
                        bigquery.SchemaField(
                            'enabled', 'BOOLEAN', mode='REQUIRED'
                        ),
                        bigquery.SchemaField(
                            'itemlabelname', 'STRING', mode='REQUIRED'
                        ),
                    ],
                ),
            ],
        ),

        bigquery.SchemaField(
            'postapplication',
            'RECORD',
            mode='NULLABLE',
            fields=[
                bigquery.SchemaField(
                    'additemlabels',
                    'RECORD',
                    mode='REPEATED',
                    fields=[
                        bigquery.SchemaField(
                            'itemlabelid', 'STRING', mode='REQUIRED'
                        ),
                        bigquery.SchemaField(
                            'description', 'STRING', mode='NULLABLE'
                        ),
                        bigquery.SchemaField(
                            'enabled', 'BOOLEAN', mode='REQUIRED'
                        ),
                        bigquery.SchemaField(
                            'itemlabelname', 'STRING', mode='REQUIRED'
                        ),
                    ],
                ),
                bigquery.SchemaField(
                    'removeitemlabels',
                    'RECORD',
                    mode='REPEATED',
                    fields=[
                        bigquery.SchemaField(
                            'itemlabelid', 'STRING', mode='REQUIRED'
                        ),
                        bigquery.SchemaField(
                            'description', 'STRING', mode='NULLABLE'
                        ),
                        bigquery.SchemaField(
                            'enabled', 'BOOLEAN', mode='REQUIRED'
                        ),
                        bigquery.SchemaField(
                            'itemlabelname', 'STRING', mode='REQUIRED'
                        ),
                    ],
                ),
            ],
        ),

        bigquery.SchemaField('event_id', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('user_created', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('user_updated', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('date_created', 'TIMESTAMP', mode='REQUIRED'),
        bigquery.SchemaField('date_updated', 'TIMESTAMP', mode='REQUIRED'),
        bigquery.SchemaField('etl_date_created', 'TIMESTAMP', mode='REQUIRED'),
        bigquery.SchemaField('etl_date_updated', 'TIMESTAMP', mode='REQUIRED'),
        bigquery.SchemaField('is_deleted', 'BOOLEAN', mode='REQUIRED')
    ]

    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    job_config.write_disposition=bigquery.WriteDisposition.WRITE_APPEND,


    if from_local_file:
        if not file_path:
            raise ValueError(
                "file_path is required when from_local_file=True"
            )

        with open(f"/tmp/{file_path}", "rb") as source_file:
            load_job = client.load_table_from_file(
                source_file,
                table_ref,
                job_config=job_config,
            )

    else:
        if not file_path:
            raise ValueError(
                "file_path is required when from_local_file=False"
            )

        file_path = (
            f"gs://{bucket_name}/"
            f"{file_path}"
        )

        load_job = client.load_table_from_uri(
            file_path,
            table_ref,
            job_config=job_config,
        )

    assert load_job.job_type == "load"

    load_job.result()

    print(f"Loaded {load_job.output_rows} rows into {dataset_id}.{table_id}")


def upload_to_gcs(
    project_id: str,
    bucket_name: str,
    local_file_path: str,
    destination_blob_name: str,
):
    client = storage.Client(project=project_id)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_file_path)

    print(
        f"Uploaded '{local_file_path}' to "
        f"gs://{bucket_name}/{destination_blob_name}"
    )

def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    args = parse_args(parser)
    config = ENVIRONMENT[args.env]
    project_id = config["project_id"]

    if args.bucket_name is not None and not args.bucket_name.strip():
        parser.error("Bucket name cannot be empty or contain only whitespace.")

    lb = args.lower_bound_timestamp
    should_insert_through_local_file = not args.bucket_name is None and not args.bucket_name.strip()
    file_name = f"{datetime.now().strftime("%Y%m%d-%H%M%S")}.jsonl"

    recipe_engine = get_engine(project_id,f'{args.env}-spanner', RECIPE_DATABASE)
    logger.info("Getting recipe data from spanner")
    bq_spanner_df = get_recipe_data_spanner(recipe_engine, lb)
    logger.info("Spanner recipe data retrieval completed")

    logger.info("Getting recipe data from BQ")
    bq_recipe_df = read_bq_to_polars(get_recipe_bq_data(project_id=project_id), None)
    logger.info("BQ recipe data retrieval completed")

    temp = bq_spanner_df.join(bq_recipe_df, on='denormalized_recipe_id', how='left')
    discrepancy = temp.filter(pl.col("recipe_id_right").is_null())
    logger.info(f"Total identified missing records {discrepancy.shape[0]}")

    columns = [
        "business_unit_id",
        "recipe_id",
        "denormalized_recipe_id",
        "recipe_payload",
    ]
    discrepancy_file_name =f"/tmp/discrepancy_{execution_time}.csv"
    discrepancy.select(columns).write_csv(discrepancy_file_name)
    logger.info(f"discrepancies stored to {discrepancy_file_name}")

    recipe_bq_insertion_list = []
    for row in discrepancy.iter_rows(named=True):
        rec_payload = row['recipe_payload']
        rec_payload_dict = json.loads(rec_payload)
        status, rec_pl = map_recipe(rec_payload_dict)
        if not status:
            continue
        recipe_bq_insertion_list.append(json.dumps(rec_pl))

    if recipe_bq_insertion_list:
        with open(f"/tmp/{file_name}", "w") as f:
            f.write("\n".join(recipe_bq_insertion_list))

        logger.info(f"JSONL load file for missing recipes stored to /tmp/{file_name}")
        if should_insert_through_local_file:
            logger.info(f"Uploading recipe jsonl file to bucket: {args.bucket_name} and file: {file_name}")
            upload_to_gcs(
                project_id=project_id,
                bucket_name=args.bucket_name,
                local_file_path=r"/tmp/" + file_name,
                destination_blob_name=file_name,
            )
            logger.info(f"GCS file upload completed")

            load_nested_json(project_id, False, file_name, bucket_name=args.bucket_name)
        else:
            logger.info("taking local jsonl file for loading into BQ")
            load_nested_json(project_id, True, file_name, None)
            logger.info("Completed")
    else:
        logger.info("Nothing to upload")


if __name__ == '__main__':

    main()