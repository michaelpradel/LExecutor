# from https://github.com/django/django/blob/main/django/forms/forms.py


def fut():
    # Collect fields from current class and remove them from attrs.
    attrs['declared_fields'] = {
        key: attrs.pop(key) for key, value in list(attrs.items())
        if isinstance(value, Field)
    }

    new_class = super().__new__(mcs, name, bases, attrs)

    # Walk through the MRO.
    declared_fields = {}
    for base in reversed(new_class.__mro__):
        # Collect fields from base class.
        if hasattr(base, 'declared_fields'):
            declared_fields.update(base.declared_fields)

        # Field shadowing.
        for attr, value in base.__dict__.items():
            if value is None and attr in declared_fields:
                declared_fields.pop(attr)

    new_class.base_fields = declared_fields
    new_class.declared_fields = declared_fields

    return new_class


if __name__ == '__main__':
    fut()
