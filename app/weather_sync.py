from app.services.weather_service import sync_weather_for_all_sites


def main():
    sync_weather_for_all_sites()


if __name__ == "__main__":
    main()
