from typing import Dict, Any

# These are sent to Gemini so it knows what functions it can call
TOOL_DEFINITIONS = [
    {
        "function_declarations": [
            {
                "name": "request_police_escort",
                "description": "Request a police escort for a citizen who feels unsafe. This will immediately alert the dashboard.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_location": {
                            "type": "string",
                            "description": "Description of where the person is (e.g., 'near central park entrance')"
                        },
                        "urgency": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "How urgent the escort request is"
                        }
                    },
                    "required": ["user_location"]
                }
            },
            {
                "name": "report_parking_violation",
                "description": "Report a vehicle parked illegally. Will log license plate if seen.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "license_plate": {"type": "string"},
                        "location": {"type": "string"}
                    }
                }
            }
        ]
    }
]

# This will be called by the server when Gemini requests a function
class FunctionHandler:
    def __init__(self, dashboard_callback):
        self.dashboard_callback = dashboard_callback  # Function to send alerts to UI
    
    async def request_police_escort(self, user_location: str, urgency: str = "medium") -> Dict:
        print(f"🚨 ESCORT REQUEST: {user_location} (urgency: {urgency})")
        # Trigger dashboard popup
        if self.dashboard_callback:
            self.dashboard_callback({
                "type": "escort_alert",
                "location": user_location,
                "urgency": urgency,
                "timestamp": "now"
            })
        return {"status": "escort_dispatched", "message": f"Police notified for escort at {user_location}"}
    
    async def report_parking_violation(self, license_plate: str = "", location: str = "") -> Dict:
        print(f"📸 PARKING VIOLATION: Plate={license_plate}, Location={location}")
        if self.dashboard_callback:
            self.dashboard_callback({
                "type": "parking_alert",
                "plate": license_plate,
                "location": location
            })
        return {"status": "logged", "ticket_id": "T12345"}