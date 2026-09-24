from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recency_days: float = Field(ge=0, description="Days since last purchase")
    frequency: float = Field(ge=0, description="Number of orders to date")
    monetary: float = Field(ge=0, description="Total revenue to date")
    tenure_days: float = Field(ge=0, description="Days since first purchase")
    avg_order_value: float = Field(ge=0)
    avg_days_between_orders: float = Field(ge=0)
    orders_30d: float = Field(ge=0)
    orders_60d: float = Field(ge=0)
    orders_90d: float = Field(ge=0)
    revenue_30d: float = Field(ge=0)
    revenue_60d: float = Field(ge=0)
    revenue_90d: float = Field(ge=0)
    distinct_products: float = Field(ge=0)
    is_uk: int = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    churn_probability: float
    predicted_label: int
    model_version: str


class CustomerPredictResponse(PredictResponse):
    customer_id: str
    as_of_date: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
